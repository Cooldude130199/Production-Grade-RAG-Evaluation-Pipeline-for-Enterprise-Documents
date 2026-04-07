from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL

def get_model():
    return SentenceTransformer(EMBED_MODEL)
