"""
rag_qdrant_pipeline.py
----------------------
Handles:
1. Loading and checking precomputed embeddings
2. Uploading them to Qdrant (if not uploaded yet)
3. Performing RAG-based medical question answering with Groq + Qdrant
"""

import os
import gc
import json
import time
import pickle
import psutil
import numpy as np
from tqdm.auto import tqdm
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


# =====================================================
# Configuration
# =====================================================

load_dotenv()

DATA_PATH = os.getenv("DATASET_PATH_JSON", "./../dataset/medical_qa_documents_with_id.json")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
QD_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "medical-faq")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "multi-qa-MiniLM-L6-cos-v1")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSIONALITY", "384"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_BATCH_SIZE=int(os.getenv("QDRANT_BATCH_SIZE","100"))

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

client = Groq(api_key=GROQ_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# =====================================================
# Data Loading
# =====================================================

def load_documents(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = [doc for group in data for doc in group.get("documents", [])]
    print(f"Loaded {len(documents)} documents from {json_path}")
    return documents


# =====================================================
# Embedding Loading or Generation
# =====================================================

def load_or_generate_embeddings(documents):
    """Loads precomputed embeddings if available, otherwise generates and saves them."""
    vec_file_pkl = f"./../dataset/documents-vectors-{EMBEDDING_MODEL_NAME}.pkl"

    if os.path.exists(vec_file_pkl):
        print(f"Found existing embeddings at {vec_file_pkl}. Loading...")
        with open(vec_file_pkl, "rb") as f:
            vectors = pickle.load(f)
        print(f"Loaded {len(vectors)} precomputed embeddings.")
    else:
        print(f"Generating new embeddings using {EMBEDDING_MODEL_NAME}...")
        vectors = []
        for doc in tqdm(documents, desc="Generating embeddings"):
            text = f"{doc.get('question', '')} {doc.get('answer', '')}".strip()
            vectors.append(model.encode(text))
        with open(vec_file_pkl, "wb") as f:
            pickle.dump(vectors, f)
        print(f"Saved new embeddings to {vec_file_pkl}")

    return vectors


# =====================================================
# Qdrant Setup and Upload
# =====================================================

def upload_to_qdrant(collection_name, vectors, documents, batch_size):
    """Uploads vectors and payloads in batches to Qdrant."""
    process = psutil.Process(os.getpid())

    for start in tqdm(range(0, len(vectors), batch_size), desc="Uploading batches"):
        end = start + batch_size
        batch_vectors = vectors[start:end]
        batch_docs = documents[start:end]

        points = [
            models.PointStruct(
                id=start + i,
                vector=np.array(vec).tolist(),
                payload=doc
            )
            for i, (vec, doc) in enumerate(zip(batch_vectors, batch_docs))
        ]

        qdrant.upsert(collection_name=collection_name, points=points)
        gc.collect()
        mem = process.memory_info().rss / 1e6
        print(f"Uploaded {end}/{len(vectors)} | Memory: {mem:.2f} MB")
        time.sleep(0.05)

    print(f"All embeddings uploaded to collection '{collection_name}'.")


def ensure_qdrant_collection(qdrant, collection_name, vectors, documents, batch_size):
    """Ensures Qdrant collection exists and is populated."""
    if qdrant.collection_exists(collection_name):
        count = qdrant.count(collection_name=collection_name).count
        if count > 0:
            print(f"Collection '{collection_name}' already exists with {count} points. Skipping upload.")
            return
        else:
            print(f"Collection '{collection_name}' exists but empty. Uploading now...")
            upload_to_qdrant(collection_name, vectors, documents, batch_size)
    else:
        print(f"Creating new collection '{collection_name}'...")
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE
            ),
        )
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name="qtype",
            field_schema="keyword"
        )
        print(f"Collection '{collection_name}' created successfully.")
        upload_to_qdrant(collection_name, vectors, documents, batch_size)


# =====================================================
# RAG Functions
# =====================================================

def build_prompt(query, search_results):
    """Builds the RAG prompt using retrieved context."""
    prompt_template = """
You are a professional medical assistant.
Answer the QUESTION using only the CONTEXT provided from verified medical sources.
If the answer is not available in the CONTEXT, say "I'm not sure based on the available information."

QUESTION: {question}

CONTEXT:
{context}
""".strip()

    context = ""
    for doc in search_results:
        context += f"question: {doc['question']}\nanswer: {doc['answer']}\n\n"

    return prompt_template.format(question=query, context=context).strip()


def llm(prompt):
    """Queries the Groq API."""
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def vector_search(query, qtype=None, limit=5):
    """Performs semantic vector search from Qdrant."""
    query_vector = model.encode(query).tolist()

    query_filter = None
    if qtype:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="qtype", match=models.MatchValue(value=qtype))]
        )

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True
    )

    return [p.payload for p in results.points]


def rag(query, qtype=None):
    """Runs the full RAG pipeline."""
    search_results = vector_search(query, qtype)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


# =====================================================
# Main Execution
# =====================================================

if __name__ == "__main__":
    print("Starting RAG Qdrant pipeline...")

    # 1. Load documents
    documents = load_documents(DATA_PATH)

    # 2. Load or generate embeddings
    vectors = load_or_generate_embeddings(documents)

    # 3. Upload embeddings to Qdrant (if needed)
    ensure_qdrant_collection(qdrant, QD_COLLECTION_NAME, vectors, documents, QDRANT_BATCH_SIZE)

    # 4. Test query
    query = "what is malaria?"
    qtype = "information"
    answer = rag(query, qtype)
    print("\nRAG Answer:\n", answer)

