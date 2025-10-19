"""
generate_embeddings.py
----------------------
Generates and saves embedding vectors for medical QA documents using a SentenceTransformer model.

Outputs:
- Pickle file: ./../dataset/documents-vectors-<model>.pkl
- JSON file:   ./../dataset/documents-vectors-<model>.json
"""

import os
import json
import pickle
from tqdm.auto import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# =====================================================
# Configuration
# =====================================================

load_dotenv()

DATA_PATH = os.getenv("DATASET_PATH_JSON", "./../dataset/medical_qa_documents_with_id.json")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
OUTPUT_DIR = os.path.dirname(DATA_PATH) or "./../dataset"

# =====================================================
# Utilities
# =====================================================

def load_documents(json_path: str):
    """Load documents from a nested JSON dataset structure."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ Dataset not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = [doc for group in data for doc in group.get("documents", [])]
    print(f"Loaded {len(documents)} documents from {json_path}")
    return documents


def save_embeddings(vectors, model_name: str, output_dir: str):
    """Save embeddings in both pickle and JSON formats."""
    base_name = model_name.split("/")[-1]
    pkl_path = os.path.join(output_dir, f"documents-vectors-{base_name}.pkl")
    json_path = os.path.join(output_dir, f"documents-vectors-{base_name}.json")

    # Save as pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(vectors, f)
    print(f"Saved embeddings (pickle): {pkl_path}")

    # Save as JSON
    vectors_json = [v.tolist() for v in vectors]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vectors_json, f)
    print(f"Saved embeddings (json): {json_path}")


# =====================================================
# Main Workflow
# =====================================================

def main():
    print("Starting embedding generation...")
    
    # Load dataset
    documents = load_documents(DATA_PATH)

    # Load model
    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Generate embeddings
    vectors = []
    for doc in tqdm(documents, desc="Generating embeddings"):
        question = doc.get("question", "")
        answer = doc.get("answer", "")
        text_to_encode = f"{question} {answer}".strip()
        vector = model.encode(text_to_encode)
        vectors.append(vector)

    # Save embeddings
    save_embeddings(vectors, EMBEDDING_MODEL, OUTPUT_DIR)

    print(f"Successfully generated {len(vectors)} embeddings.")
    print("Embedding generation completed.")


if __name__ == "__main__":
    main()



