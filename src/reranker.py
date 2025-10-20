# src/reranker.py
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from . import config

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def rerank(
    query: str, candidates: List[Dict], top_k: int = 3
) -> List[Tuple[Dict, float]]:
    """
    candidates: list of payload dicts (must contain 'answer' key)
    returns list of (payload, score) sorted descending, top_k entries
    """
    if not candidates:
        return []
    model = get_reranker()
    pairs = [[query, c.get("answer", "")] for c in candidates]
    scores = model.predict(pairs)  # higher -> better
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
