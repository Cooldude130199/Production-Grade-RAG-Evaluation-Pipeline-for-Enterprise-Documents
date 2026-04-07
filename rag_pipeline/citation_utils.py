# rag_pipeline/citation_utils.py
from rapidfuzz import fuzz

def validate_citation(text: str, citation: str, threshold: int = 80) -> bool:
    """
    Validate whether a citation appears in text.
    Uses fuzzy matching to allow near matches.
    """
    score = fuzz.partial_ratio(citation.lower(), text.lower())
    return score >= threshold
