import json

def evaluate(gold_file="ground_truth.json", retrieved_file="retrieved.json", ks=[1,3,5,10]):
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_set = json.load(f)
    with open(retrieved_file, "r", encoding="utf-8") as f:
        retrieved = json.load(f)

    print("\nEvaluation Results")
    print("k   Retrieval Accuracy   Precision   F1")

    for k in ks:
        total_queries = 0
        correct_queries = 0
        precision_scores = []
        recall_scores = []

        for g in gold_set:
            query = g["query"].strip()
            if query not in retrieved:
                continue

            retrieved_docs = set([r["doc"] + "_chunk" + str(r["chunk_id"]) for r in retrieved[query][:k]])
            gold_docs = set(g["relevant_docs"])

            # Retrieval Accuracy: at least one relevant doc in top k
            if retrieved_docs & gold_docs:
                correct_queries += 1
            total_queries += 1

            # Precision@k
            precision = len(retrieved_docs & gold_docs) / len(retrieved_docs) if retrieved_docs else 0
            precision_scores.append(precision)

            # Recall@k (for F1 calculation)
            recall = len(retrieved_docs & gold_docs) / len(gold_docs) if gold_docs else 0
            recall_scores.append(recall)

        retrieval_accuracy = correct_queries / total_queries if total_queries > 0 else 0
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0

        print(f"{k:<3} {retrieval_accuracy*100:>7.2f}%           {avg_precision*100:>7.2f}%     {f1*100:>7.2f}%")
