import os
import json
import PyPDF2
from openai import OpenAI

client = OpenAI()

# --- PDF extraction ---
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# --- Chunking function ---
def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- Batched embedding function ---
def get_openai_embeddings_batch(texts):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts  # send list of chunks at once
    )
    return [item.embedding for item in response.data]

# --- Build index ---
def build_index(docs_folder="raw_data", output_file="techm_index.json", batch_size=10):
    index = []
    for filename in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, filename)
        print(f"Indexing {filename}...")

        # Handle PDF vs text
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        # Chunk text
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        print(f"  -> {len(chunks)} chunks created")

        # Process in batches
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start+batch_size]
            embeddings = get_openai_embeddings_batch(batch)
            for i, (chunk, emb) in enumerate(zip(batch, embeddings), start=start):
                index.append({
                    "doc": filename,
                    "chunk_id": i,
                    "text": chunk,
                    "embedding": emb
                })
            print(f"    Embedded chunks {start+1}–{min(start+batch_size, len(chunks))}/{len(chunks)}")

    # Save index
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(index, f)
    print(f"✅ Index built with {len(index)} chunks and saved to {output_file}")

# --- Run ---
if __name__ == "__main__":
    build_index()
