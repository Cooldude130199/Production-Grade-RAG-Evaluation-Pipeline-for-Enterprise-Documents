import json
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI

client = OpenAI()

def get_openai_embedding(query: str):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class HybridRetriever:
    def __init__(self, index_file="techm_index.json"):
        with open(index_file, "r") as f:
            self.index = json.load(f)
        self.corpus = [entry["text"] for entry in self.index]
        self.bm25 = BM25Okapi([doc.split() for doc in self.corpus])

    def search(self, query, top_k=5):
        bm25_scores = self.bm25.get_scores(query.split())
        query_emb = get_openai_embedding(query)
        embed_scores = [cosine_similarity(query_emb, entry["embedding"]) for entry in self.index]

        scores = [(i, 0.5 * bm25_scores[i] + 0.5 * embed_scores[i]) for i in range(len(self.index))]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [self.index[i] for i, _ in scores[:top_k]]

if __name__ == "__main__":
    retriever = HybridRetriever()
    results = retriever.search("EBIT margin improvement Q3 FY25")
    for r in results:
        print(r["doc"], r["chunk_id"], r["text"][:200])
