# src/embeddings.py
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def encode_texts(texts: List[str]) -> List[List[float]]:
    """
    Encode list of texts. Returns Python lists (not numpy arrays) to be Qdrant-safe.
    """
    model = get_embedding_model()
    vecs = model.encode(texts, show_progress_bar=False)
    # ensure Python lists
    return [v.tolist() if hasattr(v, "tolist") else list(map(float, v)) for v in vecs]


def encode_text(text: str) -> List[float]:
    return encode_texts([text])[0]
