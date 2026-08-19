# 🛒 E-Commerce Valuation Engine: Baseline vs. Production Chain-of-Thought (CoT)

[![Model Architecture](https://img.shields.io/badge/Architecture-Qwen2.5--3B%20%7C%20Llama--3.2--3B-blue.svg)](https://huggingface.co/unsloth/Qwen2.5-3B-Instruct)
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
