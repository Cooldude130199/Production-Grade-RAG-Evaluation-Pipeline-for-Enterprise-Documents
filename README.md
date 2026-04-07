# RAG_Project

Production‑Grade RAG Evaluation Pipeline for Enterprise Documents

📌 Overview
This project demonstrates a production‑grade Retrieval‑Augmented Generation (RAG) evaluation pipeline built on enterprise PDFs. It automates ground truth labeling, evaluates retrieval accuracy, and reports latency and cost metrics — all with recruiter‑ready documentation and engineering discipline.

⚙️ Tech Stack
Python — indexing, auto‑labeling, evaluation scripts
Hugging Face Transformers — cross‑encoder reranker (ms-marco-MiniLM-L-12-v2)
Sentence Transformers — embeddings for query/document similarity
JSON — storage for index, ground truth, and retrieved results
VS Code / Conda Environment — CPU‑only setup for cost control
Optional Cloud (Azure/AWS/GCP) — scalable deployments and CI/CD integration

🔄 Pipeline Diagram
PDFs → Chunking → Embeddings → Index (techm_index.json)
Queries (ground_truth.json) → Auto‑Labeling → Relevant Docs
Retriever → Reranker → Evaluation (metrics.py)


📊 ASCII Pipeline Diagram

+------------------+       +------------------+       +------------------+
|   Enterprise     |       |   Ground Truth   |       |   Evaluation     |
|      PDFs        |       |   Queries (JSON) |       |   Metrics (JSON) |
+--------+---------+       +---------+--------+       +---------+--------+
         |                           |                          |
         v                           v                          v
   Chunking + Embeddings       Auto-Labeling            Retrieval Accuracy,
         |                           |                  Precision, F1, Latency, Cost
         v                           v
+------------------+       +------------------+
|   Index (JSON)   |       | Relevant Docs    |
+------------------+       +------------------+
         \___________________________/
                      |
                      v
             Retriever + Reranker
                      |
                      v
                Evaluation Script



📊 Evaluation Metrics

The pipeline reports:
Retrieval Accuracy@k — % queries with at least one relevant doc in top‑k
Precision@k — % retrieved docs that are relevant
F1 Score@k — harmonic mean of precision and recall
Latency per query — average retrieval time
Cost per query — estimated API cost

📈 Example Results (30 Queries)

Evaluation Results
k   Retrieval Accuracy   Precision   F1
1     33.33%             33.33%       16.67%
3     50.00%             23.33%       23.33%
5     56.67%             15.33%       19.17%
10    56.67%             15.33%       19.17%


Average latency per query: ~2.09s
Average cost per query: ~$0.00006
Total cost for 30 queries: ~$0.0018

🧹 Resource Hygiene
CPU‑only setup (no GPU DLL errors, cost‑controlled)
Explicit cleanup scripts for cloud resources
Cost estimates logged per query


📌 Future Work
Expand queries beyond 30 for broader coverage
Integrate CI/CD pipelines for automated evaluation
Add visual dashboards (Accuracy vs k chart, latency distribution)
Explore hybrid retrieval (BM25 + embeddings)

🎯 Conclusion
This project delivers a recruiter‑ready demo of a RAG evaluation pipeline:
Automated ground truth labeling
Retrieval Accuracy, Precision, F1 metrics
Latency and cost discipline
Modular, reproducible scripts for CI/CD
It demonstrates both technical depth and engineering discipline, making it a strong flagship project for portfolio presentation.



