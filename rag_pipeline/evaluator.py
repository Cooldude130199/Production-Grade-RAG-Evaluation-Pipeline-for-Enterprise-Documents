# evaluator.py
from sentence_transformers import util

def evaluate_citation(answer, citation_text, model, threshold=0.7):
    score = util.cos_sim(model.encode(answer), model.encode(citation_text))
    return score.item() > threshold
