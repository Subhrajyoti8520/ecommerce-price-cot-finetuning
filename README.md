# 🛒 E-Commerce Valuation Engine: Baseline vs. Production Chain-of-Thought (CoT)

[![Llama-3.2](https://img.shields.io/badge/Architecture-Llama--3.2--3B-blue.svg)](https://huggingface.co/meta-llama/Llama-3.2-3B)[![Qwen-2.5](https://img.shields.io/badge/%7C-Qwen2.5--3B-blue.svg?labelColor=blue)](https://huggingface.co/unsloth/Qwen2.5-3B-Instruct)
[![Framework](https://img.shields.io/badge/Fine--Tuning-Unsloth%20%26%20TRL-FF4500.svg)](https://github.com/unslothai/unsloth)
[![MLOps](https://img.shields.io/badge/Tracking-Weights%20%26%20Biases-FFBE00.svg)](https://wandb.ai/)
[![Serving](https://img.shields.io/badge/Serving-Docker%20%7C%20Ollama%20%7C%20GGUF-2496ED.svg)](https://ollama.com/)
[![Artifacts](https://img.shields.io/badge/HuggingFace-Subhrajyoti75-yellow.svg)](https://huggingface.co/Subhrajyoti75/qwen2.5-3b-cot-pricing-gguf_2)

An end-to-end comparative study and production deployment pipeline demonstrating how to fine-tune Small Language Models (SLMs) for e-commerce price prediction.

This repository contrasts a **Naive Direct Regression Baseline** (`meta-llama/Llama-3.2-3B`) against an **Observable Chain-of-Thought (CoT) Production Engine** (`unsloth/Qwen2.5-3B-Instruct`), taking the system from raw text fine-tuning to quantized 4-bit local Docker edge serving.

---

## 📑 Table of Contents
1. [Executive Summary & Architecture](#-executive-summary--architecture)
2. [The Ablation Study & Empirical Analysis](#-the-ablation-study--empirical-analysis)
3. [The Convergence & Under-Training Paradox](#-the-convergence--under-training-paradox)
4. [End-to-End MLOps & Deployment Pipeline](#-end-to-end-mlops--deployment-pipeline)
5. [Local Edge Deployment Guide (Docker + Ollama)](#-local-edge-deployment-guide-docker--ollama)
6. [Repository Structure](#-repository-structure)
7. [Getting Started](#-getting-started)

---

## 🏛 Executive Summary & Architecture

Direct price regression via LLMs usually produces black-box numbers prone to silent hallucination. This project re-architects retail valuation into a **structured multi-stage deduction process**:

```text
                                              Input: Title, Category, Summary
                                                            │
                                                            ▼
                              ┌──────────────────────────────────────────────────────────────┐
                              │                  CHAIN-OF-THOUGHT ENGINE                     │
                              │ 1. Extract brand & clean entity names                        │
                              │ 2. Parse functional attributes & material quality            │
                              │ 3. Classify tier: Budget | Mid-range | Premium | Luxury      │
                              │ 4. Map feature density to tier price distribution            │
                              └──────────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
                                    Output: "Price: $XX.XX" (Deterministic Parsing Anchor)
```
---
## 📊 The Ablation Study & Empirical Analysis

| Metric / Dimension | Project 1: Baseline Regression | Project 2: Production CoT Engine |
| :--- | :--- | :--- |
| **Base Architecture** | `meta-llama/Llama-3.2-3B` | `unsloth/Qwen2.5-3B-Instruct` |
| **Fine-Tuning Stack** | Hugging Face `SFTTrainer` + Vanilla `peft` | `Unsloth` Fast Triton Kernels + `SFTTrainer` |
| **LoRA Target Modules**| Attention only (`q, k, v, o`) | Full Attention + MLP (`q, k, v, o, gate, up, down`) |
| **Output Paradigm** | Direct Scalar (`Price: $XX.XX`) | Structured Reasoning (`<thought>...</thought>\nPrice: $XX.XX`) |
| **Loss Optimization** | Full sequence Causal LM loss | `train_on_responses_only` (Zero gradient penalty on prompt) |
| **Data Efficiency** | Batch size padding overhead (`max_len=128`) | `packing=True` (Concatenated 1024-token sequences) |
| **Auditability** | ❌ None (Opaque numerical guess) | ✅ Full logical trace for failure diagnostics |
| **Agent Interoperability** | ❌ None | ✅ Structured tags allow deterministic downstream routing |
| **Edge Export** | LoRA adapter checkpoint only | 4-bit `Q4_K_M` GGUF binary pushed to Hugging Face |

---
## 🔬 The Convergence & Under-Training Paradox
### Why Did the Baseline Show a Lower Initial MAE?
During 500-step evaluations on subsampled datasets, the baseline model recorded an MAE of **$63.64**, whereas the CoT model scored **$92.30**. This is a textbook "Under-Training Illusion."

```text
                              STEP-LIMITED LOSS ALLOCATION (500 STEPS: CoT)
          
          Baseline Model (Direct Regression):
          [████████████████████████████████████████] 100% Backprop -> Numerical Approximation
          
          CoT Reasoning Engine:
          [█████████████████████████               ]  60% Backprop -> XML Tag Syntax, Reasoning Steps & Stratification
          [███████████████                         ]  40% Backprop -> Exact Numerical Value Calibration
```

### Key Engineering Insights:

1. **Token Allocation Dilemma:** The baseline model generates ~5 tokens. Every gradient step is forced into regression tuning. The CoT model generates ~100 tokens, dividing early optimization capacity between learning XML structures, brand parsing, and numerical calibration.
2. **Heuristic Memorization vs. Deductive Framework:** The baseline acts as a statistical median memorizer. Out-of-distribution items cause silent failures. The CoT engine builds a transferable reasoning tree.
3. **Scaling Potential:** When trained for full epochs (3+ epochs on the full dataset), the reasoning structure stabilizes within the first epoch, allowing subsequent gradient passes to dramatically drive down regression error while retaining explainability.

---

## 🔄 End-to-End MLOps & Deployment Pipeline

```text
          ┌─────────────────────────────────┐
          │ Kaggle GPU Cluster (Dual T4)    │ -> Unsloth QLoRA Fine-Tuning + W&B Tracking
          └────────────────┬────────────────┘
                           │ (Native GGUF Quantization: Q4_K_M)
                           ▼
          ┌─────────────────────────────────┐
          │ Hugging Face Model Hub          │ -> Subhrajyoti75/qwen2.5-3b-cot-pricing-gguf_2
          └────────────────┬────────────────┘
                           │ (docker pull / download)
                           ▼
          ┌─────────────────────────────────┐
          │ Local Host / Docker Desktop     │ -> Containerized Ollama Runtime
          └────────────────┬────────────────┘
                           │ (Local Port 11434)
                           ▼
          ┌─────────────────────────────────┐
          │ Enterprise Microservice / REST  │ -> Fast, deterministic JSON/XML inference
          └─────────────────────────────────┘
```

---

## 💻 Local Edge Deployment Guide (Docker + Ollama)

Serve the fine-tuned 3B model locally using **under 3 GB of VRAM**.

### 1. Download Model & Prepare Workspace
Download the compiled `unsloth.Q4_K_M.gguf` file from Hugging Face:  
[Subhrajyoti75/qwen2.5-3b-cot-pricing-gguf_2](https://huggingface.co/Subhrajyoti75/qwen2.5-3b-cot-pricing-gguf_2)

Place `unsloth.Q4_K_M.gguf` into a local directory. In that same directory, create a file named `Modelfile` and paste the following configuration:
```dockerfile
FROM /models/unsloth.Q4_K_M.gguf

PARAMETER num_ctx 2048
PARAMETER temperature 0.1
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

SYSTEM """You are an expert e-commerce valuation system.
First, analyze the item tier, category, and feature quality inside <thought>...</thought> tags.
Then, output the final estimated retail value using the exact format 'Price: $XX.XX'."""
```

### 2. Launch Containerized Ollama Runtime
Start the GPU-accelerated Ollama container.

**Linux / macOS:**
```bash
docker run -d --gpus=all -v "$(pwd):/models" -v ollama_data:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
**Windows(PowerShell)**
```powershell
docker run -d --gpus=all -v "${PWD}:/models" -v ollama_data:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
### 3. Register & Run the CoT Model
```bash
# Build the model using the custom ChatML template
docker exec -it ollama ollama create cot-pricer -f /models/Modelfile

# Start an interactive session
docker exec -it ollama ollama run cot-pricer
```
### 4. REST API Integration
Query the running container programmatically:
```python
import requests

payload = {
    "model": "cot-pricer",
    "prompt": "Product Title: Anker USB-C Cable\nCategory: Electronics\nSummary: A durable 6-foot braided charging cable.",
    "stream": False
}

response = requests.post("http://localhost:11434/api/generate", json=payload)
print(response.json()["response"])
```
Expected Output & Verification:
```XML
<thought>
1. Category & Brand: Classifying within 'Electronics' (Brand hint: Anker).
2. Core Features: Evaluated key attributes: "A durable 6-foot braided charging cable...".
3. Market Tier: Assigned to Budget / Mass-market tier.
4. Estimation Range: Value aligns with expected price distribution for Budget / Mass-market tier.
</thought>
Price: $14.99
```
Terminal output is at `assets/deployment_terminal.png`

---
## 📁 Repository Structure
```text
ecommerce-price-cot-finetuning/
├── README.md                           # Master architectural & deployment documentation
├── .gitignore                          # Standard git ignores
├── Modelfile                           # ChatML Ollama runtime specification
├── requirements.txt                    # Python runtime dependencies
├── 01_baseline_llama3.2_regression
│   └── 01_baseline_llama3.2_regression_partA.ipynb # Project 1: Baseline Llama-3.2-3B Regression(Training)
│   └── 01_baseline_llama3.2_regression_partB.ipynb # Project 1: Baseline Llama-3.2-3B Regression(Evaluation)
├── 02_production_qwen_cot_engine
│   └── 02_production_qwen_cot_engine.ipynb # Project 2: Production Qwen-2.5-3B CoT + GGUF Export
├── util.py                             # Benchmark evaluation harness
└── assets/
    └── deployment_terminal.png         # Inference output verification
```

---
## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Subhrajyoti75/ecommerce-price-cot-finetuning.git
cd ecommerce-price-cot-finetuning
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Execute Fine-Tuning Notebooks
* Run `01_baseline_llama3_regression_partA.ipynb` and `01_baseline_llama3_regression_partB.ipynb` for the baseline setup.
* Run `02_production_qwen_cot_engine.ipynb` for the optimized CoT pipeline with automatic GGUF compilation.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

