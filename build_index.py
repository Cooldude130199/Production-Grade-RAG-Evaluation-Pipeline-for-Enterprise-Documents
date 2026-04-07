from rag_pipeline.loader import load_docs
from rag_pipeline.embedder import get_model
import faiss

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Load docs
docs = load_docs()
print("Docs loaded:", len(docs))

# Chunk docs
chunked_texts = []
for d in docs:
    chunks = chunk_text(d["text"])
    for i, chunk in enumerate(chunks):
        chunked_texts.append(f"{d['id']}_chunk{i}: {chunk}")

print("Total chunks:", len(chunked_texts))

# Embed
model = get_model()
embeddings = model.encode(chunked_texts, convert_to_numpy=True)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "techm_index.faiss")

print("Rebuilt FAISS index with", index.ntotal, "vectors")

import json

# Save chunked texts so metrics.py can load them later
with open("chunked_texts.json", "w", encoding="utf-8") as f:
    json.dump(chunked_texts, f)

print("Saved chunked_texts.json with", len(chunked_texts), "chunks")
