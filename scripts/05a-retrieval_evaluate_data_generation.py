"""
Title: Generate Medical Ground Truth Questions
Author: Muhammad Shifa
Description:
    This script randomly selects 50 medical FAQ records from the dataset
    and generates 4 patient-style questions for each using the Groq LLM API.
    It ensures diversity and relevance of questions while maintaining 
    reliable error handling, retries, and structured output.

Output: ./../dataset/search_ground-truth-data.csv
"""

import os
import json
import random
import time
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from groq import Groq


# ======================================================================
# Configuration
# ======================================================================

load_dotenv()

DATA_PATH = os.getenv("DATASET_PATH_JSON", "./../dataset/medical_qa_documents_with_id.json")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
OUTPUT_PATH = "./../dataset/search_ground-truth-data.csv"
SAMPLE_SIZE = 50

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY. Please set it in your .env file.")



client = Groq(api_key=GROQ_API_KEY)


# ======================================================================
# Functions
# ======================================================================

def load_documents(json_path: str):
    """Load documents from the provided JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found at: {json_path}")

    with open(json_path, "r") as f:
        docs_raw = json.load(f)

    documents = []
    for group in docs_raw:
        if "documents" in group:
            documents.extend(group["documents"])

    if not documents:
        raise ValueError("No documents found in dataset.")

    print(f"Loaded {len(documents)} documents from dataset.")
    return documents


def generate_questions(doc, model_name: str):
    """Generate 4 patient-style questions for a given FAQ record."""
    prompt_template = """
You are a helpful and specialized medical assistant emulating a patient.
Your task is to formulate 4 generic, clear, diverse, and natural questions 
that a patient might ask based on a medical FAQ record.
The answer of the question must and should be in medical context 
and the question should be meaningful, complete, and not too short.
If possible, use as few words as possible from the record.

The Medical FAQ Record:

Answer: {answer}
Question: {question}
Qtype: {qtype}

Output only a valid JSON list (no explanations, no code blocks):

["question1", "question2", "question3", "question4"]
""".strip()

    prompt = prompt_template.format(**doc)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def safe_json_parse(raw_json):
    """Safely parse model output into a JSON list of questions."""
    try:
        parsed = json.loads(raw_json)
        if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
            return parsed
    except Exception:
        pass
    return []


def generate_ground_truth(documents, sample_size: int, output_path: str):
    """Main routine to generate ground-truth questions and save to CSV."""
    if len(documents) < sample_size:
        raise ValueError(f"Dataset has fewer than {sample_size} documents.")

    # Skip if already generated
    if os.path.exists(output_path):
        print(f"Ground truth file already exists at: {output_path}")
        return

    doc_sample = random.sample(documents, sample_size)
    print(f"Selected {sample_size} random documents for question generation.")

    results = {}

    for doc in tqdm(doc_sample, desc="Generating questions"):
        doc_id = doc["id"]
        if doc_id in results:
            continue

        for attempt in range(3):
            try:
                json_response = generate_questions(doc, GROQ_MODEL_NAME)
                results[doc_id] = json_response
                break
            except Exception as e:
                print(f"Error for doc_id={doc_id}: {e} (attempt {attempt + 1}/3)")
                time.sleep(random.uniform(3, 6))
        else:
            print(f"Skipped doc_id={doc_id} after 3 failed attempts.")

    # Parse results
    parsed_results = {doc_id: safe_json_parse(js) for doc_id, js in results.items()}
    doc_index = {d["id"]: d for d in doc_sample}

    final_records = []
    for doc_id, questions in parsed_results.items():
        qtype = doc_index[doc_id]["qtype"]
        for q in questions:
            final_records.append((q, qtype, doc_id))

    df = pd.DataFrame(final_records, columns=["question", "qtype", "document"])
    df.to_csv(output_path, index=False)

    print(f"Ground truth CSV saved successfully at: {output_path}")
    print(f"Total generated questions: {len(df)}")


# ======================================================================
# Main Entry Point
# ======================================================================

def main():
    documents = load_documents(DATA_PATH)
    generate_ground_truth(documents, SAMPLE_SIZE, OUTPUT_PATH)


if __name__ == "__main__":
    main()

