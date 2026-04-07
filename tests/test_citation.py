# test_evaluator.py
from rag_pipeline.citation_utils import validate_citation

def test_exact_match():
    assert validate_citation("Quarterly earnings FY26", "earnings FY26")

def test_fuzzy_match():
    assert validate_citation("TechM Q2 FY26 results", "FY26 results")
