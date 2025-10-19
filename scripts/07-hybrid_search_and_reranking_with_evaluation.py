"""
hybrid_search_evaluation.py

Performs:
 - Sparse (BM25) ingestion and search in Qdrant
 - Dense (embedding) ingestion and search in Qdrant
 - Hybrid / multi-stage / RRF fusion search
 - Reranking with a Cross-Encoder
 - Evaluation (Hit Rate, MRR) for hybrid search and reranking

Behavior:
 - Uses existing embeddings if available on disk
 - Does not recreate Qdrant collections if they already exist and contain points
 - Uploads data in batches
"""

import os
import json
import uuid
import time
import gc
import pickle
from typing import List, Dict, Tuple, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Configuration (from .env)
# -------------------------
load_dotenv()

DATASET_PATH_JSON = os.getenv("DATASET_PATH_JSON", "./../dataset/medical_qa_documents_with_id.json")
GROUND_TRUTH_CSV = os.getenv("GROUND_TRUTH_PATH", "./../dataset/search_ground-truth-data.csv")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "multi-qa-MiniLM-L6-cos-v1")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSIONALITY", "384"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
SPARSE_COLLECTION = os.getenv("QDRANT_SPARSE_COLLECTION", "medical-faq-sparse")
HYBRID_COLLECTION = os.getenv("QDRANT_HYBRID_COLLECTION", "medical-faq-sparse-and-dense")
BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "200"))
EMBEDDINGS_PKL = os.getenv(
    "EMBEDDINGS_PKL",
    f"./../dataset/documents-vectors-{EMBEDDING_MODEL_NAME}.pkl"
)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


# -------------------------
# Initialize clients / models
# -------------------------
qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL)

# -------------------------
# Utilities
# -------------------------
def load_documents(json_path: str) -> List[Dict]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        docs_raw = json.load(f)
    documents = [doc for group in docs_raw for doc in group.get("documents", [])]
    return documents


def ensure_sparse_collection(collection_name: str):
    """
    Create a sparse-only collection (BM25) if it doesn't exist.
    If it exists, do nothing.
    """
    if qdrant.collection_exists(collection_name):
        count = qdrant.count(collection_name=collection_name).count
        print(f"Collection '{collection_name}' exists with {count} points.")
        return

    print(f"Creating sparse collection '{collection_name}'...")
    qdrant.create_collection(
        collection_name=collection_name,
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
    )
    # create payload index for qtype for faster filtering
    qdrant.create_payload_index(collection_name=collection_name, field_name="qtype", field_schema="keyword")
    print(f"Sparse collection '{collection_name}' created.")


def ensure_hybrid_collection(collection_name: str, embedding_dim: int):
    """
    Create hybrid (sparse + dense) collection if not exists.
    If exists, do nothing.
    """
    if qdrant.collection_exists(collection_name):
        count = qdrant.count(collection_name=collection_name).count
        print(f"Collection '{collection_name}' exists with {count} points.")
        return

    print(f"Creating hybrid collection '{collection_name}' (sparse + dense)...")
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "dense-vector": models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
    )
    qdrant.create_payload_index(collection_name=collection_name, field_name="qtype", field_schema="keyword")
    print(f"Hybrid collection '{collection_name}' created.")


def build_point(doc: Dict, dense_vector: np.ndarray = None) -> models.PointStruct:
    """
    Build a Qdrant PointStruct for a document.
    - If dense_vector is provided, attach as 'dense-vector'
    - Always attach BM25 document under 'bm25' key
    """
    vector = {}
    if dense_vector is not None:
        vector["dense-vector"] = dense_vector.tolist()

    vector["bm25"] = models.Document(text=doc.get("answer", ""), model="Qdrant/bm25")

    payload = {
        "answer": doc.get("answer", ""),
        "question": doc.get("question", ""),
        "qtype": doc.get("qtype", ""),
        "id": doc.get("id")
    }

    return models.PointStruct(id=uuid.uuid4().hex, vector=vector, payload=payload)


def batch_upsert_points(collection_name: str, points: List[models.PointStruct], batch_size: int = BATCH_SIZE):
    for i in tqdm(range(0, len(points), batch_size), desc=f"Uploading to {collection_name}"):
        batch = points[i:i + batch_size]
        qdrant.upsert(collection_name=collection_name, points=batch)
        gc.collect()
        time.sleep(0.01)


# -------------------------
# Ingestion flows
# -------------------------
def ingest_sparse_only(documents: List[Dict], collection_name: str):
    """
    Ingest documents into a sparse-only collection using BM25 document fields.
    Will skip ingestion if collection already has points.
    """
    ensure_sparse_collection(collection_name)
    count = qdrant.count(collection_name=collection_name).count
    if count > 0:
        print(f"Skipping sparse ingestion — collection '{collection_name}' already has {count} points.")
        return

    points = []
    for doc in documents:
        point = models.PointStruct(
            id=uuid.uuid4().hex,
            vector={"bm25": models.Document(text=doc.get("answer", ""), model="Qdrant/bm25")},
            payload={
                "answer": doc.get("answer", ""),
                "question": doc.get("question", ""),
                "qtype": doc.get("qtype", ""),
                "id": doc.get("id")
            }
        )
        points.append(point)

    batch_upsert_points(collection_name, points)


def ingest_hybrid(documents: List[Dict], collection_name: str, embedding_model: SentenceTransformer, embeddings_pkl: str = EMBEDDINGS_PKL):
    """
    Ingest documents into a hybrid collection (dense + sparse).
    - Uses precomputed embeddings if embeddings_pkl exists
    - Otherwise generates embeddings and saves to embeddings_pkl
    - Skips upload if collection has points
    """
    ensure_hybrid_collection(collection_name, EMBEDDING_DIM)
    count = qdrant.count(collection_name=collection_name).count
    if count > 0:
        print(f"Skipping hybrid ingestion — collection '{collection_name}' already has {count} points.")
        return

    # load or generate embeddings
    if os.path.exists(embeddings_pkl):
        print(f"Loading embeddings from {embeddings_pkl}...")
        with open(embeddings_pkl, "rb") as f:
            vectors = pickle.load(f)
    else:
        print("Generating embeddings...")
        vectors = []
        for doc in tqdm(documents, desc="Generating embeddings"):
            text = f"{doc.get('question','')} {doc.get('answer','')}".strip()
            vectors.append(embedding_model.encode(text))
        with open(embeddings_pkl, "wb") as f:
            pickle.dump(vectors, f)
        print(f"Saved embeddings to {embeddings_pkl}")

    if len(vectors) != len(documents):
        raise ValueError("Embeddings length does not match documents length")

    points = []
    for i, doc in enumerate(documents):
        dense_vector = vectors[i]
        point = models.PointStruct(
            id=uuid.uuid4().hex,
            vector={
                "dense-vector": dense_vector.tolist(),
                "bm25": models.Document(text=doc.get("answer", ""), model="Qdrant/bm25")
            },
            payload={
                "answer": doc.get("answer", ""),
                "question": doc.get("question", ""),
                "qtype": doc.get("qtype", ""),
                "id": doc.get("id")
            }
        )
        points.append(point)

    batch_upsert_points(collection_name, points)


# -------------------------
# Search methods
# -------------------------
def bm25_search(query: str, collection_name: str, limit: int = 4) -> List[models.ScoredPoint]:
    results = qdrant.query_points(
        collection_name=collection_name,
        query=models.Document(text=query, model="Qdrant/bm25"),
        using="bm25",
        limit=limit,
        with_payload=True
    )
    return results.points


def multi_stage_search(query: str, collection_name: str, embedding_model: SentenceTransformer, limit: int = 1) -> List[models.ScoredPoint]:
    results = qdrant.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=embedding_model.encode([query])[0].tolist(),
                using="dense-vector",
                limit=(10 * limit),
            ),
        ],
        query=models.Document(text=query, model="Qdrant/bm25"),
        using="bm25",
        limit=limit,
        with_payload=True,
    )
    return results.points


def rrf_search(query: str, collection_name: str, embedding_model: SentenceTransformer, limit: int = 3) -> List[models.ScoredPoint]:
    results = qdrant.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=embedding_model.encode([query])[0].tolist(),
                using="dense-vector",
                limit=(10 * limit),
            ),
            models.Prefetch(
                query=models.Document(text=query, model="Qdrant/bm25"),
                using="bm25",
                limit=(5 * limit),
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
    )
    return results.points


def rerank_results(query: str, results: List[models.ScoredPoint], top_k: int = 5) -> List[Tuple[Dict, float]]:
    # Extract candidate payloads
    candidates = [res.payload for res in results]
    pairs = [[query, c["answer"]] for c in candidates]
    # Compute rerank scores
    scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


# -------------------------
# Evaluation utilities
# -------------------------

def hit_rate(relevance_total: List[List[bool]]) -> float:
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)



def mrr(relevance_total: List[List[bool]]) -> float:
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)
    

def evaluate_rff(ground_truth: List[Dict], search_fn: Callable[[str], List[models.ScoredPoint]]) -> Dict[str, float]:
    relevance_total = []
    for q in tqdm(ground_truth, desc="Evaluating RRF"):
        doc_id = q["document"]
        results = search_fn(q["question"])
        payloads = [p.payload for p in results]
        relevance = [p["id"] == doc_id for p in payloads]
        relevance_total.append(relevance)
    return {"hit_rate": hit_rate(relevance_total), "mrr": mrr(relevance_total)}


def evaluate_rerank(ground_truth: List[Dict], hybrid_search_fn: Callable[[str], List[models.ScoredPoint]],
                    rerank_fn: Callable[[str, List[models.ScoredPoint], int], List[Tuple[Dict, float]]],
                    top_k: int = 1) -> Dict[str, float]:
    relevance_total = []
    for q in tqdm(ground_truth, desc="Evaluating Rerank"):
        query = q["question"]
        doc_id = q["document"]
        hybrid_results = hybrid_search_fn(query)
        reranked = rerank_fn(query, hybrid_results, top_k=top_k)
        payloads = [payload for payload, _ in reranked]
        relevance = [p["id"] == doc_id for p in payloads]
        relevance_total.append(relevance)
    return {"hit_rate": hit_rate(relevance_total), "mrr": mrr(relevance_total)}


# -------------------------
# Main
# -------------------------
def main():
    documents = load_documents(DATASET_PATH_JSON)
    print(f"Loaded {len(documents)} documents.")

    # Ingest sparse-only collection (if needed)
    ingest_sparse_only(documents, SPARSE_COLLECTION)

    # Ingest hybrid collection (dense + sparse)
    ingest_hybrid(documents, HYBRID_COLLECTION, embed_model, embeddings_pkl=EMBEDDINGS_PKL)

    # Load sample from ground truth
    if not os.path.exists(GROUND_TRUTH_CSV):
        raise FileNotFoundError(f"Ground truth CSV not found at: {GROUND_TRUTH_CSV}")

    df_gt = pd.read_csv(GROUND_TRUTH_CSV)
    ground_truth = df_gt.to_dict(orient="records")
    print(f"Loaded {len(ground_truth)} ground truth records.")

    # Evaluate RRF fusion search (top-5)
    rrf_metrics = evaluate_rff(ground_truth, lambda q: rrf_search(q, HYBRID_COLLECTION, embed_model, limit=5))
    print("RRF metrics:", rrf_metrics)

    # Evaluate reranking pipeline (hybrid -> rerank top 10 -> keep top 3)
    rerank_metrics = evaluate_rerank(
        ground_truth[:5],
        hybrid_search_fn=lambda q: rrf_search(q, HYBRID_COLLECTION, embed_model, limit=10),
        rerank_fn=lambda query, results, top_k=3: rerank_results(query, results, top_k=3),
        top_k=3
    )
    print("Rerank metrics (top 3):", rerank_metrics)


if __name__ == "__main__":
    main()

