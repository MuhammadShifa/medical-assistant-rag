# src/retriever.py
from typing import Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from . import config, embeddings

_qdrant = None
_embed_model = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=config.QDRANT_URL, timeout=60)
    return _qdrant


def rrf_hybrid_search(
    query: str, qtype: Optional[str] = None, limit: int = 5
) -> List[models.ScoredPoint]:
    """
    Hybrid search using Qdrant Fusion RRF:
    - Prefetch dense and bm25
    - Use Fusion.RRF to combine results
    Returns list of ScoredPoint (with .payload and .score)
    """
    qdrant = get_qdrant_client()
    embed_model = embeddings.get_embedding_model()

    query_vector = embed_model.encode([query])[0].tolist()

    prefetch_dense = models.Prefetch(
        query=query_vector, using="dense-vector", limit=limit * 10
    )
    prefetch_bm25 = models.Prefetch(
        query=models.Document(text=query, model="Qdrant/bm25"),
        using="bm25",
        limit=limit * 5,
    )

    query_filter = None
    if qtype:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="qtype", match=models.MatchValue(value=qtype))
            ]
        )

    response = qdrant.query_points(
        collection_name=config.HYBRID_COLLECTION,
        prefetch=[prefetch_dense, prefetch_bm25],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )
    return response.points


def get_payloads_from_points(points: List[models.ScoredPoint]) -> List[Dict]:
    return [p.payload for p in points]
