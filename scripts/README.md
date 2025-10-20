## 📘 File Descriptions

These are the same file of notebook but converted to python (.py) for automation. Here we have interesting logic for developing a
production based scripts. The scripts will run automatically and easily from terminal and will do their job easily.
The files print beautiful and simple terminal messages and will not repeat the tasks such as once data get indexed it will not re-index. 

**Note:**. The `.env` file should be adjusted accordingly. I will provide my own with API key and everything in details but anyone who want to run it should set their own `GROQ API`.

### **01 — Preprocessing and Parse FAQ**
Cleans and normalizes the medical dataset:
- Removes duplicates and noise.  
- Parses raw medical FAQs into structured (Question, Answer, unique id, qtypte) format.  
- Saves clean data for ingestion.

### **02 — Ingestion and RAG Flow with MinSearch**
Implements a **lightweight local search engine** (MinSearch) for quick RAG experimentation.  
Uses keyword-based retrieval to generate initial results.
query need to be adjusted within the code

### **03 — Ingestion and RAG Flow with Elasticsearch**
Builds a more scalable retrieval pipeline using **Elasticsearch**.  
Focuses on indexin in elastic search and searching in elastics search with a rag flow.
query need to be adjusted within the code


**Note:** Before testing this code Elastic Search should be run through docker.
```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "cluster.routing.allocation.disk.threshold_enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.19.4
```

### **04a — Generate Embedding Vector**
Generates dense vector embeddings using **SentenceTransformers** or similar models.  
Saves vector representations for use with vector databases in csv and json format.

### **04b — Ingestion and RAG Flow with Vector Search (Qdrant)**
Implements ingestion into **Qdrant**, a vector search database.  
Performs semantic retrieval using cosine similarity with rag pipeline.

**Note:** Before testing this code Qdrant should be run through docker.
```bash
docker run -p 6333:6333 -p 6334:6334    -v "$(pwd)/qdrant_storage:/qdrant/storage:z"    qdrant/qdrant
```


### **05a — Retrieval Evaluation Data Generation**
Creates evaluation datasets for benchmarking retrieval quality.Generate 4 question for each answers, the dataset was too much and beacuse of free trial
limitation, I have slected only 200 and generate questions for that.
Prepares ground-truth pairs for later use in retrieval metrics.

### **05b — Retrieval Evaluation (Elastic, Qdrant, and MinSearch)**
Runs retrieval evaluation across multiple backends (MinSearch, Elasticsearch, Qdrant Vector Search).  
Compares results using **Hit@K** and **Mean Reciprocal Rank (MRR)**.

### **06 — RAG Evaluator with Qdrant Vector Search**
Integrates Qdrant retrieval with **LLM-based answer generation**. It performs multiple LLM comparison as well as LLM as a Judge. 
Evaluates generated answers based on retrieved context relevance. 

### **07 — Hybrid Search and Reranking with Evaluation**
Combines **dense + sparse retrieval (Hybrid RRF)** and adds **cross-encoder reranking**.  
This module achieves the best overall performance in the pipeline. Also evaluated the hybrid search

---
