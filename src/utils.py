# src/utils.py
import json
import os
from typing import Dict, List

from . import config


def load_documents(path: str = None) -> List[Dict]:
    path = path or config.DATASET_PATH_JSON
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = [d for group in data for d in group.get("documents", [])]
    return docs


def build_prompt(question: str, contexts: List[Dict]) -> str:
    """
    Conservative prompt: instructs LLM to only use provided context.
    contexts: list of payloads {question, answer, qtype, id}
    """
    context_text = "\n\n".join(
        [
            f"question: {c.get('question','')}\nanswer: {c.get('answer','')}"
            for c in contexts
        ]
    )
    prompt = f"""
You are a professional medical assistant.
Answer the QUESTION using only the CONTEXT provided from verified medical sources.
If the answer is not available in the CONTEXT, say "I'm not sure based on the available information."

QUESTION: {question}

CONTEXT:
{context_text}
""".strip()
    return prompt
