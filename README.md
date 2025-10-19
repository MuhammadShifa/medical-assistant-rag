# 🩺  Mindra — Medical RAG System

This repository implements a **Retrieval-Augmented Generation (RAG)** system for **medical question-answering**, built on top of the [Comprehensive Medical Q&A Dataset](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset).  
The project aims to evaluate, compare, and optimize different retrieval and hybrid search techniques (BM25, Elasticsearch, MinSearch, Qdrant vector search, and reranking) for building a **trustworthy AI medical assistant**.

---

## 💡 Problem Statement

Medical professionals and patients often search for accurate and contextually relevant information from large knowledge bases. Traditional retrieval systems (like keyword search) fail to capture **semantic relationships** between questions and answers.  
This project develops a **Retrieval-Augmented Generation (RAG)** pipeline that:

1. Parses and cleans a large medical FAQ dataset.  
2. Embeds and indexes the data using multiple vector databases.  
3. Performs hybrid and reranked retrieval.  
4. Evaluates and compares search quality using metrics like Hit@K and MRR.  
5. Integrates the best-performing system into a **Streamlit-powered conversational assistant**.

---

## 📦 Dataset

**Source:** [Comprehensive Medical Q&A Dataset — Kaggle](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset)  
The dataset contains a diverse collection of **medical questions and expert answers** across multiple categories such as:
- information  
- symptoms  
- inheritance  
- causes  
- research
  etc.

It is ideal for evaluating retrieval and RAG systems because it includes structured question-answer pairs that can be semantically indexed and retrieved.

---

## 🧩 Project Structure


## ⚙️ Tech Stack

| Component        | Technology Used |
|------------------|-----------------|
| Vector Store     | Qdrant |
| Search Engine    | Elasticsearch, MinSearch, VectorSearch |
| Embeddings       | SentenceTransformers |
| LLM Integration  | Groq API (LLM Inference) |
| Reranker         | Cross-Encoder (Sentence Transformers) |
| Evaluation       | Custom metrics (Hit@K, MRR) |
| Frontend (UI)    | Streamlit |
| Environment      | conda, Python 3.10+|

---

## 🚀 Setup Instructions
