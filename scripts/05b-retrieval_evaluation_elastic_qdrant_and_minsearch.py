"""
This script performs retrieval evaluation for both Elasticsearch, qdrant and MinSearch
based on pre-generated ground truth questions.

It computes:
- Hit Rate (Recall)
- Mean Reciprocal Rank (MRR)

"""

import os
import json
import pandas as pd
import minsearch
from tqdm.auto import tqdm
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


# ============================ Configuration ============================

DATA_PATH = os.getenv("DATASET_PATH_JSON", "./../dataset/medical_qa_documents_with_id.json")
GROUND_TRUTH_PATH = os.getenv("GROUND_TRUTH_DATA", "./../dataset/search_ground-truth-data.csv")
ELASTIC_URL = os.getenv("ELASTIC_URL","http://localhost:9200")
ES_TIMEOUT = 60
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "medical-questions")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QD_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "medical-faq")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "multi-qa-MiniLM-L6-cos-v1")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSIONALITY", "384"))


es_client = Elasticsearch([ELASTIC_URL], request_timeout=ES_TIMEOUT)
qdrant_client = QdrantClient(url=QDRANT_URL, timeout=60)
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Client connected and embedding model loaded")

# ============================ Helper Functions ============================

def load_documents(path):
    """Load all medical FAQ documents from JSON file."""
    with open(path, "rt") as f_in:
        docs_raw = json.load(f_in)
    documents = []
    for group in docs_raw:
        if "documents" in group:
            documents.extend(group["documents"])
    return documents


def load_ground_truth(path):
    """Load ground-truth CSV and convert to dict records."""
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def hit_rate(relevance_total):
    """Compute Hit Rate (Recall) metric."""
    count = sum(True in line for line in relevance_total)
    return count / len(relevance_total) if relevance_total else 0.0


def mrr(relevance_total):
    """Compute Mean Reciprocal Rank (MRR) metric."""
    total_score = 0.0
    for line in relevance_total:
        for rank, rel in enumerate(line):
            if rel:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total) if relevance_total else 0.0


# ============================ Elasticsearch ============================


def elastic_search(client, query, qtype):
    """Retrieve top results from Elasticsearch for given query and qtype."""
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "answer", "qtype"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {"qtype": qtype}
                }
            }
        }
    }
    response = client.search(index=ES_INDEX_NAME, body=search_query)
    return [hit["_source"] for hit in response["hits"]["hits"]]



    
# ============================ Qdrant VectorSearch ============================
    
def vector_search(qdrant_client, query, qtype=None, limit=5):
    """Performs semantic vector search from Qdrant."""
    query_vector = model.encode(query).tolist()

    query_filter = None
    if qtype:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="qtype", match=models.MatchValue(value=qtype))]
        )

    results = qdrant_client.query_points(
        collection_name=QD_COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True
    )

    return [p.payload for p in results.points]


# ============================ MinSearch ============================

def build_minsearch_index(documents):
    """Initialize and fit MinSearch index from documents."""
    index = minsearch.Index(
        text_fields=["question", "answer"],
        keyword_fields=["qtype", "id"]
    )
    index.fit(documents)
    return index


def minsearch_search(index, query, qtype):
    """Retrieve top results using MinSearch for given query and qtype."""
    boost = {"question": 3.0}
    results = index.search(
        query=query,
        filter_dict={"qtype": qtype},
        boost_dict=boost,
        num_results=5
    )
    return results

    
# ============================ Search Evaluation ============================

def evaluate_search(ground_truth, search_function, desc='Evaluating Search'):
    relevance_total = []

    for q in tqdm(ground_truth, desc):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

# ============================ Main Evaluation Flow ============================

def main():
    print("Loading dataset and ground truth...")
    documents = load_documents(DATA_PATH)
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    print(f"Loaded {len(documents)} documents and {len(ground_truth)} ground truth records.")

    # Elasticsearch Evaluation
    print("\nRunning Elasticsearch Evaluation...")
    elastic_metrics = evaluate_search(ground_truth, lambda q: elastic_search(es_client, q['question'], q['qtype']), desc='Evaluating Elastic Search')
    print(f"Elasticsearch Results: {elastic_metrics}")
    
    # Qrant Vector Search Evaluation
    print("\nRunning Qdrant Vector Search Evaluation...")
    qdrant_metrics = evaluate_search(ground_truth, lambda q: vector_search(qdrant_client, q['question'], q['qtype']), desc='Evaluating Qdrant Search')
    print(f"Qdrant Vector Search Results: {qdrant_metrics}")

    # MinSearch Evaluation
    print("\nRunning MinSearch Evaluation...")
    min_index = build_minsearch_index(documents)
    minsearch_metrics = evaluate_search(ground_truth, lambda q: minsearch_search(min_index, q['question'], q['qtype']), desc='Evaluating MinSearch')
    print(f"MinSearch Results: {minsearch_metrics}")

    # Summary
    print("\n=== Evaluation Summary ===")
    #print(f"Elasticsearch -> Hit Rate: {elastic_metrics['hit_rate']:.4f}, MRR: {elastic_metrics['mrr']:.4f}")
    print(f"Qdrant_VectorSearch -> Hit Rate: {qdrant_metrics['hit_rate']:.4f}, MRR: {qdrant_metrics['mrr']:.4f}")
    print(f"MinSearch     -> Hit Rate: {minsearch_metrics['hit_rate']:.4f}, MRR: {minsearch_metrics['mrr']:.4f}")


if __name__ == "__main__":
    main()

