"""Public FAQ Answer Bot Trainer - Streamlit Web Application

Empowering public service administrators to fine-tune a pre-trained LLM using LoRA
to deliver verified, grounded answers to departmental frequently asked questions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app import core
from app.storage import Store

# Page configuration
st.set_page_config(
    page_title="Public FAQ Answer Bot Trainer | LoRA Studio",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "store" not in st.session_state:
    st.session_state.store = Store()
if "dataset" not in st.session_state:
    st.session_state.dataset = core.FAQDataset.from_json(core.DEFAULT_JSON_PATH)
if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt2"
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "lora_injected" not in st.session_state:
    st.session_state.lora_injected = False
if "training_history" not in st.session_state:
    st.session_state.training_history = []
if "merged_state" not in st.session_state:
    st.session_state.merged_state = False


@st.cache_resource(show_spinner="Loading Base Model & Tokenizer...")
def get_cached_base_model(model_name: str = "gpt2"):
    """Load and cache the base pre-trained model and tokenizer."""
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_name)
    return m, tok


def ensure_model_loaded():
    """Ensure session state model and tokenizer are populated."""
    if st.session_state.model is None:
        m, tok = get_cached_base_model(st.session_state.model_name)
        st.session_state.model = m
        st.session_state.tokenizer = tok


# Header Section
st.markdown('<div class="main-title">🏛️ Public FAQ Answer Bot Trainer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Train parameter-efficient LoRA adapters on departmental guidelines to answer citizen FAQs accurately without full model fine-tuning.</div>',
    unsafe_allow_html=True,
)

# Sidebar Navigation & Settings
with st.sidebar:
    st.header("⚙️ System Control")
    st.info(f"**Base Model:** {st.session_state.model_name.upper()}\n\n**Device:** {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}")

    st.markdown("---")
    st.subheader("📁 Adapter Checkpoints")
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    saved_checkpoints = sorted([p.name for p in checkpoints_dir.glob("*.pt")])

    if saved_checkpoints:
        st.write(f"**Available:** {len(saved_checkpoints)} saved adapter(s)")
        selected_checkpoint = st.selectbox("Select Checkpoint", saved_checkpoints)
        if st.button("📥 Load Selected Adapter", use_container_width=True):
            ensure_model_loaded()
            if not st.session_state.lora_injected:
                core.add_lora(st.session_state.model, r=8, alpha=16.0)
                st.session_state.lora_injected = True
            res = core.load_adapter(st.session_state.model, checkpoints_dir / selected_checkpoint)
            st.success(f"Loaded {res['loaded_modules']} adapter modules from {selected_checkpoint}!")
    else:
        st.caption("No custom checkpoints saved yet.")

    st.markdown("---")
    st.markdown(
        """
        **Project Information**
        - **Course:** `lfllm-lflora`
        - **Target:** Public Services
        - **Method:** Low-Rank Adaptation (LoRA)
        - **Framework:** PyTorch & Transformers
        """
    )


# Main Tabs
tab_dash, tab_data, tab_train, tab_eval, tab_chat, tab_adapter = st.tabs([
    "📊 Dashboard",
    "📁 Dataset Manager",
    "⚡ LoRA Training",
    "📈 Evaluation",
    "💬 FAQ Chat",
    "💾 Adapter Management",
])


# ============================================================================
# TAB 1: DASHBOARD
# ============================================================================
with tab_dash:
    st.subheader("System Overview & LoRA Status")

    ensure_model_loaded()
    param_info = core.count_parameters(st.session_state.model)
    val_report = st.session_state.dataset.validate()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Parameters", f"{param_info['total_params']:,}")
    with col2:
        st.metric("Trainable Parameters", f"{param_info['trainable_params']:,}", f"{param_info['trainable_percentage']}%")
    with col3:
        st.metric("FAQ Records", len(st.session_state.dataset))
    with col4:
        status_text = "Merged (Zero-Latency)" if st.session_state.merged_state else ("LoRA Injected" if st.session_state.lora_injected else "Base Frozen")
        st.metric("Adapter Status", status_text)

    st.markdown("### 🔍 Architectural Workflow")
    st.markdown(
        r"""
        ```
        Citizen Question ──► Prompt Formatter ──► Pretrained GPT-2 (Frozen W₀) ──┐
                                                           │                      ├──► Answer Generation
                                                  LoRA Detour: (x @ Aᵀ @ Bᵀ) ────┘
                                                  Trainable params: ~0.12%
        ```
        - **Frozen Pretrained Weights ($W_0$):** General English language representations remain preserved.
        - **Trainable LoRA Adapters ($A, B$):** Domain-specific public service knowledge injected with low rank $r=8$.
        - **Deployment Merging:** $W_{merged} = W_0 + \frac{\alpha}{r} (B \cdot A)$ for zero-latency inference without extra runtime overhead.
        """
    )


# ============================================================================
# TAB 2: DATASET MANAGER
# ============================================================================
with tab_data:
    st.subheader("Departmental FAQ Dataset")

    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Custom FAQ Dataset (JSON or CSV)",
            type=["json", "csv"],
            help="Files must contain 'question' and 'answer' columns/fields.",
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".json"):
                    data = json.load(uploaded_file)
                    items = [core.FAQItem(**row) for row in data if "question" in row and "answer" in row]
                else:
                    df = pd.read_csv(uploaded_file)
                    items = [
                        core.FAQItem(
                            question=str(r.get("question", "")),
                            answer=str(r.get("answer", "")),
                            category=str(r.get("category", "General")),
                            source=str(r.get("source", "Uploaded File")),
                        )
                        for _, r in df.iterrows()
                    ]
                st.session_state.dataset = core.FAQDataset(items)
                st.success(f"Successfully loaded {len(items)} FAQ entries from {uploaded_file.name}!")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")

    with col_up2:
        if st.button("🔄 Reset to Default Public Service FAQs"):
            st.session_state.dataset = core.FAQDataset.from_json(core.DEFAULT_JSON_PATH)
            st.rerun()

    # Validation summary
    report = st.session_state.dataset.validate()
    c1, c2, c3 = st.columns(3)
    c1.metric("Valid Records", report["valid_items"])
    c2.metric("Categories", len(report["categories"]))
    c3.metric("Duplicates Detected", report["duplicate_count"])

    # Dataset table view
    df_preview = pd.DataFrame([item.to_dict() for item in st.session_state.dataset.items])
    selected_cat = st.selectbox("Filter by Category", ["All"] + report["categories"])
    if selected_cat != "All":
        df_preview = df_preview[df_preview["category"] == selected_cat]

    st.dataframe(df_preview, use_container_width=True, hide_index=True)


# ============================================================================
# TAB 3: LORA TRAINING
# ============================================================================
with tab_train:
    st.subheader("LoRA Adapter Configuration & Training")

    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        lora_r = st.slider("LoRA Rank (r)", min_value=2, max_value=32, value=8, step=2, help="Rank of low-rank matrices A and B.")
        lora_alpha = st.slider("LoRA Alpha (α)", min_value=4.0, max_value=64.0, value=16.0, step=4.0, help="Scaling parameter alpha.")
        lora_dropout = st.slider("LoRA Dropout", min_value=0.0, max_value=0.2, value=0.05, step=0.01)

    with col_conf2:
        train_epochs = st.slider("Training Epochs", min_value=1, max_value=5, value=1)
        learning_rate = st.select_slider("Learning Rate", options=[1e-4, 3e-4, 5e-4, 1e-3, 2e-3], value=1e-3)
        batch_size = st.selectbox("Batch Size", options=[1, 2, 4], index=1, help="Keep conservative (1-2) for smooth execution on laptop CPU/GPU.")

    st.markdown("---")
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        start_train = st.button("🚀 Start LoRA Training", type="primary", use_container_width=True)

    if start_train:
        ensure_model_loaded()
        with st.spinner("Injecting LoRA adapters and running fine-tuning loop..."):
            # Fresh injection with selected hyperparameters
            if not st.session_state.lora_injected:
                core.add_lora(
                    st.session_state.model,
                    r=lora_r,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                    target_modules=["c_attn"],
                )
                st.session_state.lora_injected = True

            train_items, val_items, _ = st.session_state.dataset.train_val_test_split(seed=42)
            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()

            history = core.train_lora_model(
                st.session_state.model,
                st.session_state.tokenizer,
                train_items,
                val_items=val_items,
                epochs=train_epochs,
                lr=learning_rate,
                batch_size=batch_size,
            )
            progress_bar.progress(1.0)
            st.session_state.training_history = history.get("train_loss", [])

            # Automatically save latest trained adapter
            auto_path = checkpoints_dir / f"lora_r{lora_r}_alpha{int(lora_alpha)}.pt"
            core.save_adapter(
                st.session_state.model,
                auto_path,
                metadata={"r": lora_r, "alpha": lora_alpha, "epochs": train_epochs, "lr": learning_rate},
            )
            st.success(f"Training complete in {history.get('training_time_seconds', 0)}s! Checkpoint saved to `{auto_path.name}`.")

    if st.session_state.training_history:
        st.markdown("#### 📉 Training Loss Curve")
        loss_df = pd.DataFrame({"Epoch": range(1, len(st.session_state.training_history) + 1), "Loss": st.session_state.training_history})
        st.line_chart(loss_df.set_index("Epoch"))


# ============================================================================
# TAB 4: EVALUATION
# ============================================================================
with tab_eval:
    st.subheader("Model Evaluation: Base vs LoRA Adaptation")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### ⚖️ Parameter & Memory Efficiency")
        eval_data = {
            "Dimension": ["Trainable Parameters", "Frozen Parameters", "Storage Footprint", "Gradient Memory"],
            "Full Fine-Tuning": ["124,439,808 (100%)", "0 (0%)", "~497 MB", "~1,000 MB+"],
            "LoRA (r=8, α=16)": ["~294,912 (~0.24%)", "124,144,896 (~99.76%)", "~1.1 MB", "~2 MB"],
            "Efficiency Gain": ["~420x fewer weights", "Weights intact", "~450x smaller", "Runs on laptop"],
        }
        st.table(pd.DataFrame(eval_data))

    with col_e2:
        st.markdown("#### 🧪 Test Set Generation & Grounding")
        _, _, test_items = st.session_state.dataset.train_val_test_split(seed=42)
        test_questions = [item.question for item in test_items] if test_items else ["How do I apply for a birth certificate?"]
        selected_q = st.selectbox("Select Test Question", test_questions)

        target_item = next((it for it in test_items if it.question == selected_q), None)
        if target_item and st.button("Run Side-by-Side Comparison"):
            ensure_model_loaded()
            with st.spinner("Generating answers..."):
                lora_ans = core.generate_answer(
                    st.session_state.model,
                    st.session_state.tokenizer,
                    selected_q,
                    category=target_item.category,
                    max_new_tokens=40,
                )
                score = core.compute_grounding_score(lora_ans, target_item.answer)

            st.write(f"**Official Reference:** {target_item.answer}")
            st.write(f"**LoRA Model Answer:** {lora_ans}")
            st.metric("Guideline Grounding Score", f"{score * 100:.1f}%")


# ============================================================================
# TAB 5: FAQ CHAT
# ============================================================================
with tab_chat:
    st.subheader("💬 Citizen Public FAQ Assistant")
    st.caption("Ask questions regarding municipal policies, registrations, permits, transit, and public works.")

    user_query = st.text_input("Enter your question:", placeholder="e.g. How do I renew my commercial driver's license?")
    col_c1, col_c2 = st.columns([1, 4])
    with col_c1:
        ask_btn = st.button("Get Official Answer", type="primary", use_container_width=True)

    if ask_btn and user_query.strip():
        ensure_model_loaded()

        # Find closest matching FAQ item for grounding verification
        best_match = None
        best_score = 0.0
        for item in st.session_state.dataset.items:
            s = core.compute_grounding_score(user_query, item.question)
            if s > best_score:
                best_score = s
                best_match = item

        with st.spinner("Consulting departmental guidelines..."):
            cat = best_match.category if best_match else "General Public Service"
            bot_answer = core.generate_answer(
                st.session_state.model,
                st.session_state.tokenizer,
                user_query,
                category=cat,
                max_new_tokens=45,
            )

        st.markdown("### 📋 Generated Department Response")
        st.write(bot_answer)

        if best_match and best_score > 0.15:
            with st.expander("📌 Supporting Departmental Guideline & Verification Source", expanded=True):
                st.write(f"**Department:** {best_match.category}")
                st.write(f"**Jurisdiction / Standard:** {getattr(best_match, 'jurisdiction', 'Municipal Guideline Standard')}")
                st.write(f"**Verified Policy:** {best_match.answer}")
                st.write(f"**Official Source:** [{best_match.source}]({best_match.source})")
        else:
            st.warning("⚠️ **Department Notice:** This question may fall outside the documented guidelines in the current dataset. Please consult the municipal city clerk directly.")


# ============================================================================
# TAB 6: ADAPTER MANAGEMENT
# ============================================================================
with tab_adapter:
    st.subheader("💾 Adapter Checkpoints & Deployment Merging")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 📦 Save Active Adapter")
        new_name = st.text_input("Adapter Checkpoint Name", value="faq_adapter_v1.pt")
        if st.button("Save Current Adapter"):
            ensure_model_loaded()
            if not st.session_state.lora_injected:
                st.error("No LoRA adapter injected yet.")
            else:
                p = core.save_adapter(st.session_state.model, checkpoints_dir / new_name)
                st.success(f"Adapter saved to `{p}` ({p.stat().st_size / 1024:.1f} KB)!")

    with col_m2:
        st.markdown("#### ⚡ Zero-Latency Model Merging")
        st.write(
            """
            **Formula:** $W_{merged} = W_0 + \\frac{\\alpha}{r} (B \\cdot A)$
            
            Folds low-rank adapter matrices directly into base model weights. Eliminates LoRA branching for production inference.
            """
        )
        if not st.session_state.merged_state:
            if st.button("🔀 Merge Adapters into Base Model", type="primary"):
                ensure_model_loaded()
                merged_count = core.merge_adapters(st.session_state.model)
                st.session_state.merged_state = True
                st.success(f"Successfully merged {merged_count} LoRA modules! Latency overhead is now 0.0ms.")
                st.rerun()
        else:
            if st.button("↩️ Unmerge & Restore Base Model"):
                ensure_model_loaded()
                unmerged_count = core.unmerge_adapters(st.session_state.model)
                st.session_state.merged_state = False
                st.info(f"Unmerged {unmerged_count} LoRA modules. Restored original base weights.")
                st.rerun()
