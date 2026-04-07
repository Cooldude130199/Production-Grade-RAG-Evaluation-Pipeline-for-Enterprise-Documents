import json
import time
from rag_pipeline.retriever import HybridRetriever
from rag_pipeline.reranker import Reranker
from rag_pipeline import evaluation

# Rough cost assumptions (adjust based on your provider/pricing)
EMBED_COST_PER_1K = 0.0001   # $ per 1K tokens for embeddings
RERANK_COST_PER_QUERY = 0.00005  # $ per reranker call

def generate_retrieved(gold_file="ground_truth.json", out_file="retrieved.json", top_k=10):
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_set = json.load(f)

    retriever = HybridRetriever()
    reranker = Reranker()

    results = {}
    total_queries = len(gold_set)
    print(f"Running retrieval for {total_queries} queries...")

    total_time = 0.0
    total_cost = 0.0

    for idx, g in enumerate(gold_set, start=1):
        query = g["query"].strip()
        print(f"[{idx}/{total_queries}] Query: {query}")

        start_time = time.time()

        # Retrieve candidates
        candidates = retriever.search(query, top_k=top_k)

        # Rerank candidates
        reranked = reranker.rerank(query, candidates)

        elapsed = time.time() - start_time
        total_time += elapsed

        # Estimate cost (very rough)
        # Assume ~100 tokens per query embedding
        embed_tokens = 100
        query_cost = (embed_tokens / 1000) * EMBED_COST_PER_1K + RERANK_COST_PER_QUERY
        total_cost += query_cost

        results[query] = reranked[:5]
        print(f"  -> Retrieved {len(reranked[:5])} results in {elapsed:.2f}s (est. ${query_cost:.5f})")

    # Save results
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg_time = total_time / total_queries if total_queries > 0 else 0
    avg_cost = total_cost / total_queries if total_queries > 0 else 0

    print(f"\n✅ Retrieval complete. Saved results to {out_file}")
    print(f"Average latency per query: {avg_time:.2f}s")
    print(f"Average estimated cost per query: ${avg_cost:.5f}")
    print(f"Total estimated cost: ${total_cost:.5f}")

if __name__ == "__main__":
    generate_retrieved()
    evaluation.evaluate("ground_truth.json", "retrieved.json")
