import json
import re
import hashlib
import pandas as pd
from tqdm.auto import tqdm


# ==========================
# Configuration
# ==========================
DATA_PATH = "../dataset/medical_qa_raw.csv"
CSV_OUTPUT_PATH = "../dataset/medical_qa_with_id.csv"
JSON_OUTPUT_PATH = "../dataset/medical_qa_documents_with_id.json"


# ==========================
# Utility Functions
# ==========================
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\?", "?", text)
    text = re.sub(r"\?+", "?", text)
    return text.strip()


def remove_duplicate_qna(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Initial dataset shape: {df.shape}")
    df_clean = df.drop_duplicates(subset=["Question"], keep="first").reset_index(drop=True)
    df_clean = df_clean.drop_duplicates(subset=["Answer"], keep="first").reset_index(drop=True)
    print(f"After duplicate removal: {df_clean.shape}")
    return df_clean


def generate_document_id(row: pd.Series) -> str:
    combined = f"{row['Question']}-{row['Answer'][:15]}"
    return hashlib.md5(combined.encode()).hexdigest()[:8]


def save_json(data: list, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(data, f_out, indent=2, ensure_ascii=False)


# ==========================
# Main Pipeline
# ==========================
def main():
    df_raw = pd.read_csv(DATA_PATH)
    print("Raw dataset shape:", df_raw.shape)
    print("Columns:", df_raw.columns.tolist())
    print("Missing values:\n", df_raw.isnull().sum())

    df = df_raw.copy()

    for col in df.select_dtypes(include=["object"]).columns:
        df.loc[:, col] = df[col].apply(clean_text)

    print("Text cleaned")

    df = remove_duplicate_qna(df)
    print("Duplicates removed")

    df["id"] = df.apply(generate_document_id, axis=1)
    print("Document IDs generated")

    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Saved cleaned CSV: {CSV_OUTPUT_PATH}")

    documents = [
        {
            "question": row["Question"].strip(),
            "answer": row["Answer"].strip(),
            "qtype": row["qtype"].strip(),
            "id": row["id"]
        }
        for _, row in df.iterrows()
    ]

    final_data = [{
        "document_info": "Comprehensive Medical Q&A Dataset",
        "documents": documents
    }]

    save_json(final_data, JSON_OUTPUT_PATH)
    print(f"Saved cleaned JSON: {JSON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
