import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq
from elasticsearch import Elasticsearch


# ==========================
# Environment Setup
# ==========================
load_dotenv()

# Configuration
DATA_PATH = os.getenv("DATASET_PATH_JSON", "./../dataset/medical_qa_documents_with_id.json")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_api_key_here")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
ELASTIC_URL = os.getenv("ELASTIC_URL","http://localhost:9200")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "medical-questions")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

# Initialize clients
client = Groq(api_key=GROQ_API_KEY)
es_client = Elasticsearch([ELASTIC_URL], request_timeout=60)


# ==========================
# Utility Functions
# ==========================
def load_documents(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = [doc for group in data for doc in group.get("documents", [])]
    print(f"Loaded {len(documents)} documents from dataset.")
    return documents


def create_or_update_index(es_client, index_name: str, documents: list):
    """Create the index if missing, or index documents if empty."""
    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "question": {"type": "text"},
                "answer": {"type": "text"},
                "qtype": {"type": "keyword"},
                "id": {"type": "keyword"}
            }
        }
    }

    if es_client.indices.exists(index=index_name):
        doc_count = es_client.count(index=index_name)["count"]
        if doc_count > 0:
            print(f"Index '{index_name}' already exists with {doc_count} documents. Skipping indexing.")
            return
        else:
            print(f"Index '{index_name}' exists but contains 0 documents. Indexing now...")
    else:
        print(f"Index '{index_name}' not found. Creating and indexing documents...")
        es_client.indices.create(index=index_name, body=index_settings)
        print(f"Index '{index_name}' created successfully.")

    for doc in tqdm(documents, desc=f"Indexing into '{index_name}'"):
        es_client.index(index=index_name, id=doc["id"], document=doc)

    print(f"Indexed {len(documents)} documents into '{index_name}'.")
    print(es_client.cluster.health())
    es_client.cat.shards(index=index_name, v=True)


# ==========================
# Search and LLM
# ==========================
def elastic_search(es_client, index_name: str, query: str, qtype: str):
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
                "filter": {"term": {"qtype": qtype}}
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    return [hit["_source"] for hit in response["hits"]["hits"]]


def build_prompt(query: str, search_results: list) -> str:
    context = "\n\n".join(
        [f"question: {doc['question']}\nanswer: {doc['answer']}" for doc in search_results]
    )
    return f"""
You are a professional medical assistant.
Answer the QUESTION using only the CONTEXT provided from verified medical sources.
If the answer is not available in the CONTEXT, say "I'm not sure based on the available information."

QUESTION: {query}

CONTEXT:
{context}
""".strip()


def llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def elastic_search_rag(query: str, qtype: str, es_client, index_name: str) -> str:
    search_results = elastic_search(es_client, index_name, query, qtype)
    prompt = build_prompt(query, search_results)
    return llm(prompt)


# ==========================
# Main Entry
# ==========================
def main():
    documents = load_documents(DATA_PATH)
    create_or_update_index(es_client, ES_INDEX_NAME, documents)

    query = "what is malaria?"
    qtype = "information"
    answer = elastic_search_rag(query, qtype, es_client, ES_INDEX_NAME)

    print(f"\nQuery: {query}")
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()

