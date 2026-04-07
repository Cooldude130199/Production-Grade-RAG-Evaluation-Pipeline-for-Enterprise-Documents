import json
import numpy as np
from openai import OpenAI

client = OpenAI()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def auto_label_ground_truth(index_file="techm_index.json",
                            query_file="ground_truth.json",
                            out_file="ground_truth.json",
                            top_n=3):
    # Load your index (chunks + embeddings)
    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    # Load queries from existing ground_truth.json template
    with open(query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    ground_truth = []
    for q in queries:
        query_text = q["query"]
        print(f"\nProcessing query: {query_text}")

        # Embed the query
        q_emb = client.embeddings.create(
            model="text-embedding-3-large",
            input=query_text
        ).data[0].embedding

        # Compute similarity with all chunks
        scored = []
        for item in index:
            emb = item["embedding"]
            sim = cosine_similarity(np.array(q_emb), np.array(emb))
            scored.append((sim, item["doc"] + "_chunk" + str(item["chunk_id"])))

        # Sort by similarity and take top_n
        scored.sort(reverse=True, key=lambda x: x[0])
        relevant = [doc_id for _, doc_id in scored[:top_n]]

        ground_truth.append({
            "query": query_text,
            "relevant_docs": relevant
        })

        print(f"  -> Top {top_n} relevant docs: {relevant}")

    # Save auto-labeled ground truth
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Auto-labeled ground truth saved to {out_file}")

if __name__ == "__main__":
    auto_label_ground_truth()
