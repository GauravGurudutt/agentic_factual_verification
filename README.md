# An Agentic Framework for Reliable Retrieval-Augmented Text Generation using Verification, Correction, and Multi-Model Evaluation

**Author:** Gaurav G  
**Institution:** PES University, Bengaluru  

---

## Overview

This repository implements a 10-agent LangGraph pipeline for factual question answering over the HotPotQA dataset. The framework addresses hallucination in Retrieval-Augmented Generation (RAG) systems through independent verification, iterative correction, and multi-model evaluation across four language model backends.

The pipeline is evaluated against the DePaC baseline and demonstrates improvements in Exact Match, F1 Score, and hallucination reduction across all four model configurations.

---

## Pipeline Architecture

The pipeline consists of 10 sequential agents:

| # | Agent | Model | Role |
|---|-------|-------|------|
| ① | QueryDecompositionAgent | LLM (model-specific) | Decomposes multi-hop questions into 2 atomic sub-questions |
| ② | HybridRetrievalAgent | BGE-large + BM25 | Dense + sparse retrieval with FAISS fusion |
| ③ | RetrievalQualityGateAgent | BGE-large | Quality scoring with Wikipedia/web fallback |
| ④ | RerankingAgent | MS-MARCO CrossEncoder | Reranks candidates by relevance |
| ⑤ | GraphRAGAgent | None (graph traversal) | BFS hop-2 entity co-occurrence expansion |
| ⑥ | ICAAgent | TF-IDF | Information-Calibrated Aggregation document scoring |
| ⑦ | SentenceSelectionAgent | TF-IDF | Extracts top-10 relevant sentences |
| ⑧ | AnswerGenerationAgent | LLM (model-specific) | N=5 self-consistency candidates with BGE tie-break |
| ⑨ | VerificationAgent | DeBERTa-v3-NLI | FEVER-inspired entailment verification (independent) |
| ⑩ | RegenerationAgent | LLM (model-specific) | Triggered by DeBERTa contradiction — stricter re-generation |

**Key design principle:** The verification agent (DeBERTa) is architecturally independent from all generation models, preventing circular self-verification.

---

## Results on HotPotQA (100 samples)

| Model | Exact Match | F1 Score | Hallucination | Avg Latency |
|-------|:-----------:|:--------:|:-------------:|:-----------:|
| Base Paper (DePaC) | 0.3095 | 0.4510 | 32.6% | — |
| Mistral-7B + QLoRA | 0.4000 | 0.5023 | 16.3% | 53.6s |
| Llama-3.1-8B + QLoRA | 0.3200 | 0.4548 | 15.0% | 8.96s |
| Gemini 3.1 Flash Lite | **0.5300** | **0.6202** | 16.0% | 52.1s |
| OpenAI GPT-4o-mini | 0.4300 | 0.5562 | **10.0%** | 13.9s |

All models outperform the DePaC baseline on hallucination rate. OpenAI achieves the lowest hallucination (10.0%) while Gemini achieves the highest EM and F1.

---

## Repository Structure

```
├── AgentRAG_OpenAI.ipynb       # Pipeline with OpenAI gpt-4o-mini
├── AgentRAG_Gemini.ipynb       # Pipeline with Gemini 3.1 Flash Lite
├── AgentRAG_Llama.ipynb        # Pipeline with Llama-3.1-8B + QLoRA
├── AgentRAG_Mistral.ipynb      # Pipeline with Mistral-7B + QLoRA
├── requirements.txt            # Python dependencies
└── README.md
```

Each notebook is self-contained and includes:
- Full 10-agent LangGraph pipeline
- Model-specific generation functions
- Flask backend with live progress streaming
- HTML frontend with HITL (Human-in-the-Loop) checkpoints
- HotPotQA evaluation with EM, F1, hallucination metrics

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/GauravGurudutt/agentic_factual_verification.git
cd agentic_factual_verification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API keys

Open the notebook you want to run and fill in Cell 1 and Cell 2:

```python
# Cell 1
HF_TOKEN = "YOUR_HF_TOKEN_HERE"          # huggingface.co/settings/tokens

# Cell 2
TAVILY_API_KEY = "YOUR_TAVILY_API_KEY_HERE"   # tavily.com

# OpenAI notebook only
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"   # platform.openai.com

# Gemini notebook only
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"   # aistudio.google.com

# Flask cell
NGROK_TOKEN = "YOUR_NGROK_TOKEN_HERE"         # dashboard.ngrok.com
```

### 4. Run on Google Colab (recommended)

Each notebook is optimised for Colab with A100/T4 GPU. Open directly:

- [AgentRAG_OpenAI.ipynb](https://colab.research.google.com/github/GauravGurudutt/agentic_factual_verification/blob/main/AgentRAG_OpenAI.ipynb) *(update link after creating repo)*
- [AgentRAG_Gemini.ipynb](https://colab.research.google.com/github/GauravGurudutt/agentic_factual_verification/blob/main/AgentRAG_Gemini.ipynb)
- [AgentRAG_Llama.ipynb](https://colab.research.google.com/github/GauravGurudutt/agentic_factual_verification/blob/main/AgentRAG_Llama.ipynb)
- [AgentRAG_Mistral.ipynb](https://colab.research.google.com/github/GauravGurudutt/agentic_factual_verification/blob/main/AgentRAG_Mistral.ipynb)

Run all cells sequentially. The evaluation cell will run 100 HotPotQA samples and print EM, F1, and hallucination metrics.

---

## Frontend

Each notebook includes a Flask backend and HTML frontend served via ngrok. After running the Flask cell, open the ngrok URL to access the interactive UI which supports:

- Live agent progress rendering
- Persona selection (Encyclopedic / Concise / Academic)
- HITL checkpoint intervention at decomposition, generation, and verification stages
- Gold answer comparison with colour-coded match indicators

---

## Requirements

```
torch
transformers>=4.40.0
sentence-transformers
faiss-cpu
rank-bm25
langgraph
flask
flask-cors
pyngrok
datasets
peft
bitsandbytes
openai
google-generativeai
tavily-python
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{gauravg2025agenticrag,
  title  = {An Agentic Framework for Reliable Retrieval-Augmented Text Generation
             using Verification, Correction, and Multi-Model Evaluation},
  author = {Gaurav G},
  year   = {2025},
  school = {PES University, Bengaluru}
}
```

---

## Acknowledgements

- [DePaC](https://arxiv.org/abs/2310.07039) — baseline framework and ICA scoring methodology
- [HotPotQA](https://hotpotqa.github.io/) — evaluation dataset
- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [BGE-large](https://huggingface.co/BAAI/bge-large-en) — dense retrieval embeddings
- [DeBERTa-v3](https://huggingface.co/cross-encoder/nli-deberta-v3-base) — NLI verification
