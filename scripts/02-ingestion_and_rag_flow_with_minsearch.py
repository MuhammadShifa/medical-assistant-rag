import os
import json
import minsearch
from dotenv import load_dotenv
from groq import Groq


# Load environment variables from .env file
load_dotenv()

# ==========================
# Configuration
# ==========================

DATA_PATH = os.getenv("DATASET_PATH_JSON", "./../dataset/medical_qa_documents_with_id.json")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_api_key_here")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME","openai/gpt-oss-20b")


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


# ==========================
# Data Loading
# ==========================
def load_documents(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = [doc for group in data for doc in group.get("documents", [])]
    print(f"Loaded {len(documents)} documents.")
    return documents


# ==========================
# Search Index Setup
# ==========================
def build_index(documents):
    index = minsearch.Index(
        text_fields=["question", "answer"],
        keyword_fields=["qtype"]
    )
    index.fit(documents)
    print("Search index built successfully.")
    return index


# ==========================
# Search and Prompt Building
# ==========================
def search(index, query: str, qtype_filter: str = "information", top_k: int = 5):
    boost = {"question": 3.0}
    return index.search(
        query=query,
        filter_dict={"qtype": qtype_filter},
        boost_dict=boost,
        num_results=top_k
    )


def build_prompt(query: str, search_results: list[str]) -> str:
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


# ==========================
# LLM Inference
# ==========================
def llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# ==========================
# RAG Pipeline
# ==========================
def rag_pipeline(query: str, qtype: str, index):
    search_results = search(index=index, query=query, qtype_filter=qtype)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


# ==========================
# Main Entry
# ==========================
def main():
    documents = load_documents(DATA_PATH)
    index = build_index(documents)

    query = "what is malaria?"
    qtype = 'information'
    answer = rag_pipeline(query, qtype, index)

    print(f"\nQuery: {query}")
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()
