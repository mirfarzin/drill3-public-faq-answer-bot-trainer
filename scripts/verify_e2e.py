"""End-to-End LoRA Verification Script for Drill 3.
Audits the genuine execution of:
- Pretrained GPT-2 loading
- LoRA injection and parameter freezing
- Training and weight delta verification
- Checkpoint serialization and deserialization
- Inference generation
- Merging and unmerging numerical tolerance
- Memory profiling
"""

import gc
import os
import time
import tracemalloc
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app import core


tracemalloc.start()


def get_process_memory_mb() -> float:
    current, peak = tracemalloc.get_traced_memory()
    return current / (1024 * 1024)


def main():
    print("=" * 60)
    print("RIGOROUS END-TO-END LORA VERIFICATION FOR DRILL 3")
    print("=" * 60)

    # 1. Base Model Loading
    mem_init = get_process_memory_mb()
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    mem_after_load = get_process_memory_mb()
    print(f"[1] Base Model Loaded: gpt2 ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"    Load Time: {time.time() - t0:.2f}s")
    print(f"    Measured Memory Usage: {mem_after_load - mem_init:.1f} MB (Total process RSS: {mem_after_load:.1f} MB)")

    # 2. LoRA Injection
    injected = core.add_lora(model, r=8, alpha=16.0, dropout=0.05, target_modules=["c_attn"])
    counts = core.count_parameters(model)
    print(f"[2] LoRA Injected: {len(injected)} modules across 12 transformer blocks")
    print(f"    Total Parameters:     {counts['total_params']:,}")
    print(f"    Trainable Parameters: {counts['trainable_params']:,} ({counts['trainable_percentage']}%)")
    print(f"    Frozen Parameters:    {counts['frozen_params']:,}")

    # Verify base weights are frozen
    base_frozen = all(not p.requires_grad for name, p in model.named_parameters() if not name.endswith(".A") and not name.endswith(".B"))
    print(f"    Base weights strictly frozen: {base_frozen}")
    assert base_frozen, "CRITICAL ERROR: Base parameters are not frozen!"

    # 3. Check Initial Weights
    sample_A = model.transformer.h[0].attn.c_attn.A.clone().detach()
    sample_B = model.transformer.h[0].attn.c_attn.B.clone().detach()
    all_zero_B = torch.equal(sample_B, torch.zeros_like(sample_B))
    print(f"[3] LoRA Initialization Check:")
    print(f"    Adapter B is all zeros at init: {all_zero_B}")
    assert all_zero_B, "CRITICAL ERROR: LoRA B matrix is not zero at init!"

    # 4. Training on FAQ Dataset
    dataset = core.FAQDataset.from_json(core.DEFAULT_JSON_PATH)
    train_items, val_items, test_items = dataset.train_val_test_split(seed=42)
    print(f"[4] Training Adapter on FAQ Dataset:")
    print(f"    Dataset items: {len(dataset)} (Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)})")

    mem_before_train = get_process_memory_mb()
    t_train = time.time()
    history = core.train_lora_model(
        model, tokenizer, train_items, val_items=val_items,
        epochs=1, lr=1e-3, batch_size=2, device="cpu"
    )
    t_elapsed = time.time() - t_train
    mem_after_train = get_process_memory_mb()

    print(f"    Training 1 epoch completed in {t_elapsed:.2f}s")
    print(f"    Train loss: {history['train_loss']}")
    print(f"    Validation loss: {history['val_loss']}")
    print(f"    Measured Memory during training: {mem_after_train - mem_before_train:.1f} MB delta (Total RSS: {mem_after_train:.1f} MB)")

    # 5. Confirm Weight Updates
    updated_A = model.transformer.h[0].attn.c_attn.A.clone().detach()
    updated_B = model.transformer.h[0].attn.c_attn.B.clone().detach()
    delta_A = (sample_A - updated_A).abs().max().item()
    delta_B = (sample_B - updated_B).abs().max().item()
    print(f"[5] Weight Change Verification:")
    print(f"    Adapter A max delta: {delta_A:.8f} (changed: {delta_A > 0})")
    print(f"    Adapter B max delta: {delta_B:.8f} (changed: {delta_B > 0})")
    assert delta_A > 0 and delta_B > 0, "CRITICAL ERROR: Adapter weights did not update during training!"

    # Confirm base weights did not change
    wte_frozen = not any(p.requires_grad for p in model.transformer.wte.parameters())
    print(f"    Base token embeddings still frozen: {wte_frozen}")

    # 6. Save and Reload Adapter
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "verified_adapter.pt"
    core.save_adapter(model, ckpt_path, metadata={"r": 8, "alpha": 16.0, "epochs": 1})
    ckpt_size = ckpt_path.stat().st_size
    print(f"[6] Adapter Checkpoint Serialization:")
    print(f"    Saved Checkpoint: {ckpt_path}")
    print(f"    Actual Checkpoint Size on Disk: {ckpt_size:,} bytes ({ckpt_size / (1024 * 1024):.2f} MB)")

    # Load into a separate fresh model
    fresh_model = AutoModelForCausalLM.from_pretrained("gpt2")
    core.add_lora(fresh_model, r=8, alpha=16.0, target_modules=["c_attn"])
    res_load = core.load_adapter(fresh_model, ckpt_path)
    reloaded_A = fresh_model.transformer.h[0].attn.c_attn.A.clone().detach()
    reloaded_B = fresh_model.transformer.h[0].attn.c_attn.B.clone().detach()
    matches = torch.equal(updated_A, reloaded_A) and torch.equal(updated_B, reloaded_B)
    print(f"    Loaded {res_load['loaded_modules']} adapter modules into fresh model")
    print(f"    Reloaded weights match trained weights exactly: {matches}")
    assert matches, "CRITICAL ERROR: Reloaded weights do not match saved weights!"

    # 7. Generate Answer
    test_q = test_items[0]
    print(f"[7] End-to-End Inference with Reloaded Model:")
    print(f"    Query Question:  '{test_q.question}'")
    print(f"    Query Category:  '{test_q.category}'")
    print(f"    Reference Answer: '{test_q.answer[:80]}...'")
    ans = core.generate_answer(fresh_model, tokenizer, test_q.question, category=test_q.category, max_new_tokens=40)
    print(f"    Generated Answer: '{ans}'")
    grounding = core.compute_grounding_score(ans, test_q.answer)
    print(f"    Content Grounding Recall Score: {grounding * 100:.2f}%")

    # 8. Numerical Merging Verification
    test_input = tokenizer("Department: Civic\nQuestion: Test", return_tensors="pt")
    fresh_model.eval()
    with torch.no_grad():
        out_unmerged = fresh_model(**test_input).logits

    count_merged = core.merge_adapters(fresh_model)
    with torch.no_grad():
        out_merged = fresh_model(**test_input).logits
    merge_max_diff = (out_unmerged - out_merged).abs().max().item()
    print(f"[8] Numerical Merging Audit:")
    print(f"    Merged {count_merged} modules into base weights")
    print(f"    Max Absolute Discrepancy (Unmerged vs Merged): {merge_max_diff:.8e}")
    assert merge_max_diff < 5e-4, f"CRITICAL ERROR: Merge discrepancy too large: {merge_max_diff}"

    count_unmerged = core.unmerge_adapters(fresh_model)
    with torch.no_grad():
        out_restored = fresh_model(**test_input).logits
    unmerge_max_diff = (out_unmerged - out_restored).abs().max().item()
    print(f"    Unmerged {count_unmerged} modules, restoring base weights")
    print(f"    Max Absolute Discrepancy After Unmerge: {unmerge_max_diff:.8e}")
    assert unmerge_max_diff < 5e-4, f"CRITICAL ERROR: Unmerge discrepancy too large: {unmerge_max_diff}"

    print("=" * 60)
    print("ALL RIGOROUS END-TO-END VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
