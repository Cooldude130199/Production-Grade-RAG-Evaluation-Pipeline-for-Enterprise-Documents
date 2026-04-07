import os
from PyPDF2 import PdfReader
from networkx import has_path
from openai import files
from .config import DATA_PATH

def load_docs(base_path=DATA_PATH):
    docs = []
    for root, _, files in os.walk(base_path):
        for fname in files:
            if fname.endswith(".pdf"):
                reader = PdfReader(os.path.join(root, fname))
                text = " ".join([page.extract_text() or "" for page in reader.pages])
                docs.append({"id": fname, "text": text})
    print(f"Loaded {len(docs)} documents")
    return docs


