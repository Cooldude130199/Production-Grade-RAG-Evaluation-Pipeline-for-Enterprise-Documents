# tests/test_indexer.py
from rag_pipeline.indexer import build_index, load_index
import numpy as np

def test_indexer_build_and_load(tmp_path):
    embeddings = np.random.rand(5, 10).astype("float32")
    index = build_index(embeddings)
    loaded = load_index()
    assert loaded.ntotal == 5
