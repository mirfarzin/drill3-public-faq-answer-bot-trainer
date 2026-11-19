# Public FAQ Answer Bot Trainer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/3hq45dsypf/badge.svg)](https://ask.thili.ai/b/3hq45dsypf)

This project develops a Public FAQ Answer Bot Trainer, empowering public service administrators to train a language model using LoRA. The goal is to provide accurate answers to frequently asked questions based on specific departmental guidelines, thereby reducing inquiry load. The solution demonstrates the application of LoRA fine-tuning to adapt a pre-trained LLM for domain-specific question answering.

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

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the dataset with a wider range of public service FAQs and answers.
- Implement more rigorous quantitative evaluation metrics for answer accuracy and relevance.
- Allow administrators to easily update the FAQ dataset and retrain the model through the interface.
