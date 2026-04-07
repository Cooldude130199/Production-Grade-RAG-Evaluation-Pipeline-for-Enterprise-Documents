# RAG_Project

# 📚 Production‑Grade RAG Evaluation Pipeline for Enterprise Documents

## 🔎 Overview
A modular **Retrieval‑Augmented Generation (RAG) evaluation pipeline** that indexes enterprise PDFs, auto‑labels ground truth, and evaluates retrievers with **accuracy, precision, latency, and cost metrics**.  
It showcases recruiter‑ready engineering discipline with **citation enforcement, CI/CD‑friendly scripts, and explicit resource hygiene**.

---

## ⚙️ Tech Stack
- 🐍 **Python** — indexing, auto‑labeling, evaluation scripts  
- 🤗 **Hugging Face Transformers** — cross‑encoder reranker (`ms-marco-MiniLM-L-12-v2`)  
- 🔎 **Sentence Transformers** — embeddings for query/document similarity  
- 📄 **JSON** — storage for index, ground truth, and retrieved results  
- 💻 **VS Code / Conda Environment** — CPU‑only setup for cost control  
- ☁️ **Optional Cloud (Azure/AWS/GCP)** — scalable deployments and CI/CD integration  

---

## 🔄 Pipeline Flow

## 📊 Pipeline Diagram

```mermaid
flowchart TD
    A[Enterprise PDFs] --> B[Chunking + Embeddings]
    B --> C[Index (JSON)]
    D[Ground Truth Queries (JSON)] --> E[Auto-Labeling]
    E --> F[Relevant Docs]
    C --> G[Retriever + Reranker]
    F --> G
    G --> H[Evaluation Script]
    H --> I[Evaluation Metrics: Accuracy, Precision, F1, Latency, Cost]





## 📊 Evaluation Results

| k   | Retrieval Accuracy | Precision | F1   |
|-----|--------------------|-----------|------|
| 1   | 33.33%             | 33.33%    | 16.67% |
| 3   | 50.00%             | 23.33%    | 23.33% |
| 5   | 56.67%             | 15.33%    | 19.17% |
| 10  | 56.67%             | 15.33%    | 19.17% |

**Latency & Cost**
- ⏱️ Avg latency: **2.09s/query**  
- 💰 Avg cost: **$0.00006/query**  
- 💵 Total cost (30 queries): **$0.0018**



## 🧹 Resource Hygiene
- ✅ CPU‑only setup (no GPU DLL errors, cost‑controlled)
- ✅ Explicit cleanup scripts for cloud resources
- ✅ Cost estimates logged per query


## 📌 Future Work
- 🔄 Expand queries beyond 30 for broader coverage
- ⚡ Integrate CI/CD pipelines for automated evaluation
- 📊 Add visual dashboards (Accuracy vs k chart, latency distribution)
- 🔍 Explore hybrid retrieval (BM25 + embeddings)

## 🎯 Conclusion

This project delivers a recruiter‑ready demo of a RAG evaluation pipeline:

- Automated ground truth labeling
- Retrieval Accuracy, Precision, F1 metrics
- Latency and cost discipline
- Modular, reproducible scripts for CI/CD