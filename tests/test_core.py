"""Comprehensive Test Suite for Public FAQ Answer Bot Trainer

Tests each of the 5 Milestones required by askThili Drill 3:
1. Dataset loading, schema validation, prompt formatting, and train/val/test splits.
2. LoRA adapter injection, parameter freezing, backprop weight updates, saving and loading.
3. Streamlit application structure and page components.
4. Qualitative and quantitative grounding evaluation, and base vs LoRA comparison.
5. Zero-latency adapter weight merging and reversible unmerging.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

from app import core


# ============================================================================
# MILESTONE 1: DATA PREPARATION & FORMATTING
# ============================================================================

def test_milestone_1():
    """Verify dataset loading, validation, splitting, and formatting."""
    result = core.milestone_1()
    assert result["status"] == "success"
    assert result["total_records"] >= 10
    assert result["train_records"] > 0
    assert result["val_records"] > 0
    assert result["test_records"] > 0
    assert len(result["categories"]) >= 3
    assert "Question:" in result["sample_prompt"]
    assert "Answer:" in result["sample_prompt"]


def test_faq_dataset_validation_detects_malformed_and_duplicates(tmp_path):
    """Test that FAQDataset correctly flags empty fields and duplicate questions."""
    raw_data = [
        {"question": "How to get a permit?", "answer": "Apply online.", "category": "Permits"},
        {"question": "How to get a permit?", "answer": "Duplicate question.", "category": "Permits"},
        {"question": "", "answer": "Missing question.", "category": "Errors"},
        {"question": "Valid Question?", "answer": "", "category": "Errors"},
    ]
    json_file = tmp_path / "test_faqs.json"
    import json
    json_file.write_text(json.dumps(raw_data), encoding="utf-8")

    ds = core.FAQDataset.from_json(json_file)
    val = ds.validate()
    assert val["duplicate_count"] == 1
    assert val["valid_items"] == 1


# ============================================================================
# MILESTONE 2: LORA ADAPTER INJECTION AND TRAINING
# ============================================================================

def test_milestone_2():
    """Verify LoRA injection, parameter freezing, weight update, and serialization."""
    res = core.milestone_2()
    assert res["status"] == "success"
    assert res["injected_layers_count"] > 0
    assert res["base_frozen"] is True
    assert res["adapter_updated"] is True
    assert res["loaded_adapter_modules"] > 0
    assert res["trainable_pct"] < 2.0  # LoRA trains < 2% of parameters


def test_lora_conv1d_forward_and_freezing():
    """Verify custom LoRAConv1D preserves base outputs at init and freezes base weights."""
    base_conv = Conv1D(2304, 768)
    lora = core.LoRALinear(base_conv, r=4, alpha=8.0)

    # Base parameters must be completely frozen
    assert not any(p.requires_grad for p in base_conv.parameters())
    # Adapter parameters must be trainable
    assert lora.A.requires_grad is True
    assert lora.B.requires_grad is True

    # At initialization (B=0), output must match base output exactly
    x = torch.randn(2, 8, 768)
    base_out = base_conv(x)
    lora_out = lora(x)
    assert torch.allclose(base_out, lora_out, atol=1e-5)


# ============================================================================
# MILESTONE 3: STREAMLIT INTERFACE
# ============================================================================

def test_milestone_3():
    """Verify Streamlit entrypoint contains all required operational views."""
    res = core.milestone_3()
    assert res["status"] == "success"
    assert res["valid"] is True
    assert len(res["missing_keywords"]) == 0


def test_main_py_compiles():
    """Verify that app/main.py is syntactically valid Python."""
    import py_compile
    py_compile.compile("app/main.py", doraise=True)


# ============================================================================
# MILESTONE 4: ACCURACY AND RELEVANCE EVALUATION
# ============================================================================

def test_milestone_4():
    """Verify generation evaluation and full fine-tuning vs LoRA comparison."""
    res = core.milestone_4()
    assert res["status"] == "success"
    assert "question" in res
    assert "reference_answer" in res
    assert "base_model_answer" in res
    assert "lora_model_answer" in res
    assert res["comparison"]["parameter_reduction_ratio"] > 100.0


def test_grounding_score_calculation():
    """Verify content token overlap recall calculation."""
    ref = "Submit Form BC-1 online via the municipal citizen portal with $15 fee."
    gen_perfect = "submit form bc-1 online via municipal citizen portal fee $15"
    gen_irrelevant = "The weather today is sunny and clear in the city."

    score_high = core.compute_grounding_score(gen_perfect, ref)
    score_low = core.compute_grounding_score(gen_irrelevant, ref)
    assert score_high > 0.5
    assert score_low == 0.0


# ============================================================================
# MILESTONE 5: ADAPTER MERGING FOR INFERENCE
# ============================================================================

def test_milestone_5():
    """Verify adapter merging produces identical outputs and is fully reversible."""
    res = core.milestone_5()
    assert res["status"] == "success"
    assert res["merged_layers"] > 0
    assert res["numerically_identical"] is True
    assert res["fully_reversible"] is True
