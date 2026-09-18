# EvidenceRAG

**Evidence-grounded Retrieval-Augmented Generation (RAG) system**

EvidenceRAG is an AI research assistant designed to generate answers grounded in retrieved evidence from user-provided documents.

The project focuses on combining **semantic search, RAG, source citation, claim extraction, and evidence verification** to improve the traceability and reliability of LLM-generated answers.

## 🎯 Project Goal

Large Language Models can generate fluent answers but may produce unsupported or incorrect information.

EvidenceRAG aims to address this problem by retrieving relevant evidence from documents and connecting generated claims to their supporting sources.

The main idea is:

```text
Documents
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Search
    ↓
Relevant Evidence
    ↓
LLM
    ↓
Generated Answer
    ↓
Claim Extraction
    ↓
Evidence Verification
```

## 🚧 Project Status

**Early Development**

Currently working on:

* Understanding embeddings
* Semantic search
* FAISS-based vector retrieval

## 🛠️ Planned Features

* [ ] Document ingestion
* [ ] PDF text extraction
* [ ] Text chunking
* [ ] Semantic embeddings
* [ ] FAISS vector search
* [ ] Retrieval-Augmented Generation (RAG)
* [ ] Source citation
* [ ] Claim extraction
* [ ] Evidence verification
* [ ] "Insufficient evidence" detection
* [ ] Retrieval evaluation
* [ ] Answer faithfulness evaluation
* [ ] FastAPI backend
* [ ] User interface
* [ ] Dockerization

## 🧰 Tech Stack

* Python
* Sentence Transformers
* FAISS
* PyTorch
* Hugging Face Transformers
* FastAPI
* Docker

## 📁 Project Structure

```text
evidenceRAG/
│
├── app/
│   ├── rag/
│   ├── models/
│   ├── evaluation/
│   └── utils/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── experiments/
├── tests/
├── notebooks/
│
├── main.py
├── requirements.txt
└── README.md
```

## 🗺️ Roadmap

### Phase 1 — Semantic Retrieval

* Embedding generation
* Vector representation
* Similarity search
* FAISS integration

### Phase 2 — RAG

* PDF processing
* Chunking
* Retrieval
* LLM-based answer generation

### Phase 3 — Evidence & Citations

* Document metadata
* Source tracking
* Citation generation

### Phase 4 — Claim Verification

* Claim extraction
* Evidence retrieval
* Claim-evidence matching
* Unsupported claim detection

### Phase 5 — Evaluation

* Precision@K
* Recall@K
* MRR
* Answer relevance
* Faithfulness
* Citation correctness

### Phase 6 — Application

* FastAPI
* User interface
* Docker
* Deployment

## 📌 Objective

The goal of EvidenceRAG is not only to generate answers, but to make the relationship between **answers, claims, and supporting evidence** explicit and measurable.
