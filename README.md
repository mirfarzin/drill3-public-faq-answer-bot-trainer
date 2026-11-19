# Public FAQ Answer Bot Trainer

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/3hq45dsypf/badge.svg)](https://msrit.ask.thili.ai/b/3hq45dsypf)

A parameter-efficient fine-tuning (PEFT) system that trains and deploys Low-Rank Adaptation (LoRA) adapters on top of a frozen pre-trained autoregressive language model (GPT-2). Designed for public service and municipal administrations, this application grounds language generation in verified departmental policies to provide accurate, hallucination-resistant answers to citizen frequently asked questions.

---

## What you are building

**Goal.** You will have built a Public FAQ Answer Bot Trainer that uses LoRA to fine-tune a language model, enabling public service administrators to provide accurate answers to frequently asked questions.

**Done when.** The project is done when the LoRA fine-tuning technique has been successfully applied to adapt a pre-trained LLM, and the bot generates accurate and relevant FAQ answers based on specific departmental guidelines, verifiable through a Streamlit interface.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Create a small dataset of public service FAQs and their answers.**  
      Hint: Data preparation & formatting
- [x] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: LoRA adapter injection & training (`lfllm-lflora`)
- [x] **3. Build a Streamlit interface where users can ask questions and get answers.**  
      Hint: Streamlit UI development
- [x] **4. Verify the accuracy and relevance of the generated answers.**  
      Hint: Qualitative evaluation
- [x] **5. Implement adapter merging for inference.**  
      Hint: Adapter merging (`lfllm-lflora`)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [The Mathematics of LoRA](#the-mathematics-of-lora)
4. [System Architecture & Workflow](#system-architecture--workflow)
5. [Dataset Format & Provenance](#dataset-format--provenance)
6. [Full Fine-Tuning vs LoRA Comparison](#full-fine-tuning-vs-lora-comparison)
7. [Installation & Setup](#installation--setup)
8. [Running the Application](#running-the-application)
9. [Running Tests & Verification](#running-tests--verification)
10. [LoRA Training & Adapter Lifecycle](#lora-training--adapter-lifecycle)
11. [Adapter Merging for Production Inference](#adapter-merging-for-production-inference)
12. [Results, Evaluation & Limitations](#results-evaluation--limitations)
13. [Viva / Demonstration Cheat-Sheet](#viva--demonstration-cheat-sheet)

---

## Project Overview

Public agencies field thousands of citizen inquiries daily regarding permits, property registrations, transit concessions, waste management, and public health guidelines. Generic foundation LLMs frequently hallucinate policies, invent statutory deadlines, or generate vague continuations. Full fine-tuning of 100M+ parameter models requires dedicated GPU clusters, risks catastrophic forgetting of general language capabilities, and incurs massive storage costs per department.

**Public FAQ Answer Bot Trainer** solves this by implementing custom **Low-Rank Adaptation (LoRA)** from scratch:
- The base model's 124.4 million weights remain **100% frozen**.
- Trainable rank decomposition matrices ($A$ and $B$) are injected into attention projection layers ($c\_attn$).
- Only **0.12% – 0.24% of parameters** are trained (~147K – 294K parameters).
- Adapters require **< 1.2 MB of storage** per departmental configuration.
- Adapters can be **merged directly into base weights** at runtime ($W_{merged} = W_0 + \frac{\alpha}{r} B \cdot A$), eliminating inference branching overhead.

---

## Problem Statement

Traditional adaptation strategies face a severe trilemma:
1. **Prompt Engineering / Few-Shot In-Context Learning:** Consumes precious context window length, lacks deterministic formatting adherence, and incurs token costs on every call.
2. **Full Fine-Tuning:** Updates all parameters ($\nabla W$). Requires large GPU VRAM for optimizer states (AdamW tracks 2 additional states per parameter, tripling memory), risks catastrophic forgetting of pre-trained syntactic fluency, and outputs a massive 500 MB+ checkpoint for every department.
3. **Low-Rank Adaptation (LoRA):** Hypothesizes that the weight updates during task-specific adaptation have a low "intrinsic dimension" ($r \ll d$). By decomposing $\Delta W$ into the product of two low-rank matrices, memory, compute, and storage footprints drop by over 400x.

---

## The Mathematics of LoRA

In a transformer block, an attention projection layer performs:
$$h = x \cdot W_0$$
where $W_0 \in \mathbb{R}^{d_{in} \times d_{out}}$ (or $W_0^T$ in standard linear formulations).

### 1. Decomposition
LoRA parameterizes the update $\Delta W$ as:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (A \cdot B)$$

Where:
- $W_0$ is frozen: $\frac{\partial \mathcal{L}}{\partial W_0}$ is never accumulated (`requires_grad = False`).
- $A \in \mathbb{R}^{d_{in} \times r}$ is initialized from a Gaussian distribution: $\mathcal{N}\left(0, \frac{1}{r}\right)$.
- $B \in \mathbb{R}^{r \times d_{out}}$ is initialized to **zeros**: $B = \mathbf{0}$.
- $r$ is the intrinsic rank ($r \ll \min(d_{in}, d_{out})$, e.g., $r = 4$ or $r = 8$).
- $\alpha$ is a constant scaling hyperparameter (e.g., $\alpha = 16$).
- Scaling factor: $\gamma = \frac{\alpha}{r}$.

### 2. Forward Pass at Initialization ($t=0$)
Because $B = \mathbf{0}$:
$$\Delta W = \frac{\alpha}{r} (A \cdot \mathbf{0}) = \mathbf{0}$$
$$\text{output} = x \cdot W_0 + \mathbf{0} = x \cdot W_0$$
**At step 0, the model's behavior is mathematically identical to the base pre-trained model.** No random noise is introduced at the start of training.

### 3. Forward Pass During Adaptation
$$h = \text{base\_layer}(x) + \left(\text{Dropout}(x) \cdot A \cdot B\right) \times \frac{\alpha}{r}$$

### 4. Adapter Merging for Zero-Latency Production Deployment
When deploying to high-throughput production environments:
$$W_{merged} = W_0 + \frac{\alpha}{r} (A \cdot B)$$
The low-rank delta is accumulated in-place directly into the base weights. The forward pass becomes a single matrix multiplication without any branch divergence or extra kernel launches:
$$h = x \cdot W_{merged}$$

---

## System Architecture & Workflow

```
                        ┌──────────────────────────────────────────────────────────┐
                        │             Public FAQ Answer Bot Trainer                │
                        └──────────────────────────────────────────────────────────┘
                                                     │
               ┌─────────────────────────────────────┴─────────────────────────────────────┐
               ▼                                                                           ▼
   [ 📁 Dataset Manager ]                                                      [ ⚡ LoRA Engine ]
   - Validate Schema                                                           - Freeze Base GPT-2 (124M)
   - Detect Missing/Duplicates                                                 - Inject LoRAConv1D on c_attn
   - Stratified Split (70/15/15)                                               - Trainable: ~0.12% params
               │                                                                           │
               └─────────────────────────────────────┬─────────────────────────────────────┘
                                                     ▼
                                       [ 📊 Training Pipeline ]
                                       - AdamW Optimizer (lr=1e-3)
                                       - Causal LM Cross-Entropy Loss
                                       - Step/Epoch Loss Tracking
                                                     │
               ┌─────────────────────────────────────┴─────────────────────────────────────┐
               ▼                                                                           ▼
   [ 📈 Evaluation Suite ]                                                     [ 💬 Citizen FAQ Chat ]
   - Perplexity & Cross-Entropy                                                - Grounded generation
   - Lexical Grounding Overlap                                                 - Supporting policy citation
   - Base vs LoRA Side-by-Side                                                 - Out-of-domain disclaimers
               │                                                                           │
               └─────────────────────────────────────┬─────────────────────────────────────┘
                                                     ▼
                                       [ 💾 Adapter Management ]
                                       - Export lightweight checkpoint (< 1.2 MB)
                                       - Fold weights: W_merged = W_0 + (alpha/r)*B*A
                                       - 0.0ms inference latency overhead
```

---

## Dataset Format & Provenance

The system operates on structured departmental FAQ records in JSON or CSV format:

| Field | Type | Description | Example |
|---|---|---|---|
| `question` | string | Citizen inquiry | "How do I apply for a certified birth certificate?" |
| `answer` | string | Official departmental policy | "Submit Form BC-1 online via the citizen portal with $15 fee..." |
| `category` | string | Department classification | "Civic Registration & Records" |
| `source` | string | Official statutory URL | `https://gov.services.local/vital-records/birth-certificates` |

### Prompt Formatting for Causal Language Modeling
During tokenization, each FAQ record is converted into an autoregressive prompt terminated by `<|endoftext|>`:
```
Department: Civic Registration & Records
Question: How do I apply for a certified copy of a birth certificate?
Answer: To obtain a certified copy of a birth certificate, submit Form BC-1 online via the municipal citizen portal or in person at the Vital Statistics Registrar office...<|endoftext|>
```

During inference, the prompt is formatted up to `Answer:`, allowing the model to generate the authoritative departmental answer.

---

## Full Fine-Tuning vs LoRA Comparison

| Metric | Full Fine-Tuning | LoRA ($r=8, \alpha=16$) | Efficiency Advantage |
|---|---|---|---|
| **Base Model** | GPT-2 (small) | GPT-2 (small) | Standard baseline |
| **Total Parameters** | 124,439,808 | 124,439,808 | Identical base model |
| **Trainable Parameters** | 124,439,808 (100%) | 294,912 (~0.24%) | **~422x fewer trainable weights** |
| **Frozen Parameters** | 0 (0%) | 124,144,896 (~99.76%) | Preserves general linguistic knowledge |
| **Optimizer Memory** | ~1,000 MB+ | ~2.3 MB | Fits comfortably in laptop RAM/VRAM |
| **Checkpoint Storage** | ~497.7 MB | ~1.1 MB | **~452x storage reduction** |
| **Catastrophic Forgetting** | High risk | Minimal risk | Base weights remain immutable |
| **Deployment Overhead** | Single model per domain | Base model shared + tiny swap | Supports multi-tenant adapters |
| **Inference Latency** | Baseline $1.0\times$ | Merged: $1.0\times$ (Zero-overhead) | No runtime latency penalty |

---

## Installation & Setup

### 1. Prerequisites
- Python 3.12 (64-bit)
- Git

### 2. Environment Setup
```bash
# Navigate to the project directory
cd drill3code/public-faq-answer-bot-trainer

# (Optional) Create and activate a virtual environment
python3.12 -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit web application:
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

### Interface Tabs
1. **📊 Dashboard:** Real-time parameter breakdown (trainable vs frozen), active adapter status, and architecture schematic.
2. **📁 Dataset Manager:** Upload custom FAQ CSV/JSON datasets, validate records against duplicates and schema errors, preview entries, and filter by department.
3. **⚡ LoRA Training:** Configure rank $r$ ($2-32$), alpha $\alpha$ ($4-64$), dropout, learning rate, and epochs. Execute fine-tuning with live training curves.
4. **📈 Evaluation:** Quantitative grounding recall scores, validation loss, side-by-side comparison between Base GPT-2 and LoRA-adapted GPT-2.
5. **💬 FAQ Chat:** Citizen query interface that retrieves relevant departmental guidelines, produces grounded answers, cites official URLs, and handles out-of-domain inquiries.
6. **💾 Adapter Management:** Save and load named checkpoints, inspect checkpoint file sizes, and merge adapters directly into base weights.

---

## Running Tests & Verification

The project includes an end-to-end automated test suite covering all 5 milestones:
```bash
pytest -v
```

### Verified Test Cases:
- `test_milestone_1`: Verifies dataset parsing, schema validation, prompt formatting, and train/val/test splits.
- `test_faq_dataset_validation_detects_malformed_and_duplicates`: Verifies duplicate question detection and empty record handling.
- `test_milestone_2`: Verifies LoRA injection, base model parameter freezing, backprop weight updates, and adapter serialization.
- `test_lora_conv1d_forward_and_freezing`: Verifies mathematical equivalence at $t=0$ ($B=\mathbf{0}$) and parameter freezing.
- `test_milestone_3`: Verifies Streamlit interface entrypoint completeness and component structure.
- `test_main_py_compiles`: Verifies `app/main.py` is syntax-valid Python.
- `test_milestone_4`: Verifies qualitative and quantitative answer evaluation and comparative metrics.
- `test_grounding_score_calculation`: Verifies token overlap recall metric and stop-word filtering.
- `test_milestone_5`: Verifies adapter weight merging ($W_{merged} = W_0 + \frac{\alpha}{r} B \cdot A$), output numerical identity within float tolerance, and reversibility.
- `test_the_store_keeps_what_you_put_in_it`: Verifies key-value state persistence.
- `test_model_settings_default_to_a_local_ollama`: Verifies environment configuration fallbacks.
- `test_the_page_file_is_valid_python`: Verifies Streamlit page compilation.

---

## Results, Evaluation & Limitations

### 1. Empirical Findings
- **Base GPT-2:** When prompted with `"Department: Civic Registration\nQuestion: How do I apply for a birth certificate?\nAnswer:"`, the pre-trained base model tends to hallucinate generic, unrelated text or repeat the prompt.
- **LoRA Model:** After parameter-efficient fine-tuning on departmental data, the model correctly outputs authoritative instructions referencing specific forms (e.g., `"Form BC-1"`), statutory search fees (`"$15"`), and submission locations (`"Vital Statistics Registrar office"`).
- **Grounding Score:** Evaluates content token overlap between generated answer and official departmental reference, filtering out stop-words.

### 2. Limitations
- **Model Capacity:** Base GPT-2 (124M) is a compact autoregressive model chosen to run locally on standard laptops without GPU requirements. For complex multi-turn reasoning, larger backbones (e.g., Llama-3-8B or Mistral-7B) can be adapted using the exact same LoRA architecture.
- **Dataset Size:** The included dataset contains 20 curated public service questions. Administrators can upload larger custom departmental datasets via the Dataset Manager.

---

## Viva / Demonstration Cheat-Sheet

If asked to explain your project in a viva or interview:
1. **What is LoRA?**  
   *LoRA stands for Low-Rank Adaptation. Instead of fine-tuning all 124 million parameters of GPT-2, we freeze the original weight matrices and inject two small trainable matrices $A$ and $B$ into each attention projection layer such that $\Delta W = \frac{\alpha}{r} (A \cdot B)$.*
2. **Why initialize $B$ to zero?**  
   *Because at step 0 of training, $\Delta W = A \cdot \mathbf{0} = \mathbf{0}$. This guarantees that the adapted model starts with the exact behavior of the pre-trained foundation model, avoiding initial random noise.*
3. **What is the purpose of adapter merging?**  
   *In production, computing $x \cdot W_0 + x \cdot A \cdot B$ introduces two sequential matrix multiplications and latency overhead. By computing $W_{merged} = W_0 + \frac{\alpha}{r} (A \cdot B)$ once at deployment time, we get zero inference latency overhead.*
4. **How much memory did you save?**  
   *We reduced trainable parameters from 124.4 million down to under 295,000 (a ~420x reduction) and checkpoint storage from ~497 MB to ~1.1 MB.*
