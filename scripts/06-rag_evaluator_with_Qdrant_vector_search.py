"""
RAG_Evaluation.py
-----------------
This script performs Retrieval-Augmented Generation (RAG) evaluation using
multiple LLMs (OSS-20B, OSS-120B, LLaMA-70B). It evaluates:
  1. Generated answers' semantic similarity for ground truth question with ground truth answers.
  2. LLM-as-a-Judge evaluation for relevance classification.

Steps:
  - Retrieve top documents from Qdrant vector DB.
  - Construct a contextual prompt.
  - Generate answers using multiple LLMs.
  - Compute cosine similarity between generated and original answers.
  - Use LLM-as-a-Judge to evaluate relevance (RELEVANT / PARTLY / NON_RELEVANT).

Outputs:
  - Cosine similarity CSVs for each model.
  - Evaluation CSV with LLM-judge classifications.
"""

import os
import time
import json
import random
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client import QdrantClient, models
from groq import Groq
import matplotlib.pyplot as plt
import seaborn as sns


# ============================ Configuration ============================
load_dotenv()

DATA_PATH = os.getenv("DATASET_PATH_JSON", "./../dataset/medical_qa_documents_with_id.json")
GROUND_TRUTH_PATH = os.getenv("GROUND_TRUTH_DATA", "./../dataset/search_ground-truth-data.csv")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QD_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "medical-faq")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "multi-qa-MiniLM-L6-cos-v1")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSIONALITY", "384"))
SAMPLE_SIZE = 2

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

client = Groq(api_key=GROQ_API_KEY)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
qdrant_client = QdrantClient(url=QDRANT_URL, timeout=60)

# ============================ Vector Search ============================

def vector_search(qdrant_client, embedding_model, query, qtype=None, limit=2):
    """Perform semantic vector search from Qdrant."""
    query_vector = embedding_model.encode(query).tolist()

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


# ============================ Prompt Construction ============================

def build_prompt(query, search_results):
    """Construct RAG prompt for LLM generation."""
    prompt_template = """
You are a professional medical assistant.
Answer the QUESTION using only the CONTEXT provided from verified medical sources.
If the answer is not available in the CONTEXT, say "I'm not sure based on the available information."

QUESTION: {question}

CONTEXT:
{context}
""".strip()

    context = "\n".join([f"question: {d['question']}\nanswer: {d['answer']}\n" for d in search_results])
    return prompt_template.format(question=query, context=context)


# ============================ LLM Generation ============================

def llm(prompt, model):
    """Generate an answer using Groq LLM API."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def rag(query, qdrant_client, embedding_model, gpt_model):
    """End-to-end RAG pipeline: retrieve, build prompt, and generate answer."""
    search_results = vector_search(qdrant_client, embedding_model, query["question"], query["qtype"])
    prompt = build_prompt(query["question"], search_results)
    return llm(prompt, gpt_model)


# ============================ Evaluation Helpers ============================

def compute_similarity(embedding_model, answer_orig, answer_llm):
    """Compute cosine similarity between embeddings."""
    emb_orig = embedding_model.encode(answer_orig)
    emb_llm = embedding_model.encode(answer_llm)
    return cosine_similarity([emb_orig], [emb_llm])[0][0]


def run_model_evaluation(gpt_model, embedding_model, gt_sampled_data, doc_idx, qd_client, output_file):
    """Run RAG + similarity evaluation for a specific model."""
    results = {}

    for i, rec in enumerate(tqdm(gt_sampled_data, desc=f"Evaluating {gpt_model}")):
        if i in results:
            continue
        try:
            answer_llm = rag(rec, qd_client, embedding_model, gpt_model)
            doc_id = rec["document"]
            answer_orig = doc_idx[doc_id]["answer"]

            similarity = compute_similarity(embedding_model, answer_orig, answer_llm)

            results[i] = {
                "answer_llm": answer_llm,
                "answer_orig": answer_orig,
                "document": doc_id,
                "question": rec["question"],
                "qtype": rec["qtype"],
                "Cosine Similarity": similarity,
            }
        except Exception as e:
            print(f"Error at index {i}: {e}")
        time.sleep(10)

    df = pd.DataFrame(results).T
    df.to_csv(output_file, index=False)
    mean_score = float(df["Cosine Similarity"].mean())
    print(f"{gpt_model} Average Cosine Similarity: {mean_score:.3f}")
    return df


# ============================ LLM-as-a-Judge ============================

prompt1_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer compared to the original answer provided.
Based on the relevance and similarity of the generated answer to the original answer, you will classify
it as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Original Answer: {answer_orig}
Generated Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the original
answer and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


prompt2_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def llm_judge_evaluation(df_results, template):
    """Evaluate relevance using LLM-as-a-Judge."""
    records = df_results.to_dict(orient="records")
    evaluations = []

    for i, record in enumerate(tqdm(records, desc="LLM-as-a-Judge")):
        try:
            prompt = template.format(**record)
            evaluation = llm(prompt, model="openai/gpt-oss-20b")
            evaluations.append(evaluation)
        except Exception as e:
            print(f"Error at index {i}: {e}")
        time.sleep(10)

    # Parse JSON safely
    parsed_evals = []
    for e in evaluations:
        try:
            parsed = json.loads(e)
            parsed_evals.append(parsed)
        except Exception:
            continue

    df_eval = pd.DataFrame(parsed_evals)
    print(df_eval["Relevance"].value_counts())
    return df_eval


# ============================ Main Execution ============================

if __name__ == "__main__":
    # Load data
    with open(DATA_PATH, "r") as f:
        docs_raw = json.load(f)
    df_ground_truth = pd.read_csv(GROUND_TRUTH_PATH)

    ground_truth = df_ground_truth.to_dict(orient="records")
    documents = docs_raw[0]["documents"]
    doc_idx = {d["id"]: d for d in documents}

    gt_sampled_data = random.sample(ground_truth, SAMPLE_SIZE)

    # Model evaluations
    df_20b = run_model_evaluation("openai/gpt-oss-20b", embedding_model, gt_sampled_data, doc_idx, qdrant_client, "rag_eval_results_20b.csv")
    df_120b = run_model_evaluation("openai/gpt-oss-120b",embedding_model, gt_sampled_data, doc_idx, qdrant_client, "rag_eval_results_120b.csv")
    df_70b = run_model_evaluation("llama-3.3-70b-versatile", embedding_model, gt_sampled_data, embedding_model, doc_idx, qdrant_client, "rag_eval_results_llama70b.csv")

    # Plot cosine similarity distributions
    sns.kdeplot(df_20b["Cosine Similarity"], label="OSS-20B")
    sns.kdeplot(df_120b["Cosine Similarity"], label="OSS-120B")
    sns.kdeplot(df_70b["Cosine Similarity"], label="LLaMA-70B")
    plt.title("RAG Model Cosine Similarity Comparison")
    plt.xlabel("Cosine Similarity")
    plt.legend()
    plt.show()

    # Run LLM-as-a-Judge
    print("\n--- LLM-as-a-Judge: Comparing Generated vs Original Answers ---")
    df_eval_1 = llm_judge_evaluation(df_20b, prompt1_template)

    print("\n--- LLM-as-a-Judge: Question-Answer Relevance ---")
    df_eval_2 = llm_judge_evaluation(df_20b, prompt2_template)

