# main_rag_pipeline.py
import os
import re
import time
from openai import OpenAI
from rag_pipeline.loader import load_docs
from rag_pipeline.embedder import get_model
from rag_pipeline.indexer import build_index, load_index
from rag_pipeline.retriever import search_index
from rag_pipeline.reranker import rerank
from rag_pipeline.citation_utils import validate_citation
from rag_pipeline.utils import log

client = OpenAI(api_key=os.getenv("sk-proj-....."))

def validate_citations(output, retrieved_ids):
    citations = re.findall(r"\[source: (.*?)\]", output)
    valid = all(c in retrieved_ids for c in citations)
    coverage = len(citations) / max(1, len(retrieved_ids))
    log(f"Citation coverage: {coverage*100:.2f}% | Valid: {valid}")
    return valid, coverage

def evaluate_retrieval(test_queries, docs, model, index):
    correct = 0
    for query, expected_doc in test_queries.items():
        results = search_index(query, docs, model, index)
        top_ids = [doc["id"] for doc in results[:3]]
        if expected_doc in top_ids:
            correct += 1
    accuracy = correct / len(test_queries) * 100
    log(f"Retrieval Accuracy: {accuracy:.2f}%")
    return accuracy

def run_pipeline(query):
    start = time.time()
    docs = load_docs()
    model = get_model()
    embeddings = model.encode([doc["text"] for doc in docs])

    try:
        index = load_index()
    except:
        index = build_index(embeddings)

    candidates = search_index(query, docs, model, index)
    reranked = rerank(query, candidates)

    citations = []
    for doc in reranked[:3]:
        if validate_citation(doc["text"], query):
            citations.append(doc["id"])

    context = "\n\n".join([doc["text"] for doc in reranked[:3]])
    citation_str = ", ".join(citations)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"""
        Summarize Tech Mahindra's quarterly earnings FY26.
        Every factual statement must include a citation like [source: {citation_str}].
        Keep the answer concise (3–4 sentences).

        Context:
        {context}
        """}],
        max_tokens=300,
        temperature=0.7
    )

    llm_output = response.choices[0].message.content.strip()

    usage = response.usage
    input_cost = usage.prompt_tokens / 1000 * 0.0015
    output_cost = usage.completion_tokens / 1000 * 0.002
    total_cost = input_cost + output_cost

    latency = time.time() - start
    log(f"Latency: {latency:.2f}s | Cost: ${total_cost:.4f}")

    validate_citations(llm_output, [doc["id"] for doc in reranked[:3]])

    return llm_output, latency, total_cost
