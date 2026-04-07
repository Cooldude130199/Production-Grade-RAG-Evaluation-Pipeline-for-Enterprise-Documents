from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.reranker = CrossEncoder(model_name)

    def rerank(self, query, candidates):
        texts = [c["text"] for c in candidates]
        scores = self.reranker.predict([(query, t) for t in texts])
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in reranked]

if __name__ == "__main__":
    # Example usage
    candidates = [
        {"doc": "sample.pdf", "chunk_id": 0, "text": "EBIT margin improved in Q3 FY25 due to Project Fortius."},
        {"doc": "sample.pdf", "chunk_id": 1, "text": "Revenue growth was strong in Q1 FY26."}
    ]
    reranker = Reranker()
    reranked = reranker.rerank("EBIT margin improvement Q3 FY25", candidates)
    for r in reranked:
        print(r["doc"], r["chunk_id"], r["text"])
