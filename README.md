# RAG Experimental Framework

## Overview
**RAG Experimental Framework** is a research-grade Retrieval-Augmented Generation (RAG) system designed to compare multiple RAG pipelines and identify the optimal configuration for specific document and query sets.

Large Language Models (LLMs) often struggle to reliably answer questions based on long documents. They may hallucinate, miss critical details, or rely heavily on pre-training data rather than the provided context. **This project addresses these limitations by treating RAG design as a controlled experiment.**

---

## Why RAG?
Retrieval-Augmented Generation improves LLM responses by grounding them in external, verifiable documents. The standard workflow used in this framework includes:

1.  **Extract:** Ingest text from documents (PDF / TXT).
2.  **Split:** Divide text into meaningful chunks.
3.  **Embed:** Convert chunks into vector embeddings.
4.  **Retrieve:** Find the most relevant chunks for a specific query.
5.  **Generate:** Produce answers strictly from the retrieved context.

[Image of Retrieval Augmented Generation architecture diagram]

This process ensures that the resulting answers are accurate, verifiable, and free from hallucinations.

---

## What This Project Does
This framework moves beyond a "one-size-fits-all" approach. It builds multiple RAG pipelines simultaneously to test different variables:

* **Embedding Models:** Tests different ways of representing text vectors.
* **Chunk Sizes:** Tests different granularities of text splitting.

### The Experiment Process
1.  **Ingest:** Loads the document.
2.  **Permutate:** Creates multiple pipelines with different configurations.
3.  **Run:** Executes the same user query across all pipelines.
4.  **Evaluate:** Scores each pipeline using objective metrics.
5.  **Select:** Automatically identifies the best-performing pipeline.

### Outputs
* The final, most accurate answer.
* Detailed evaluation metrics.
* A comprehensive pipeline comparison report.

---

## Key Features
* **PDF Document Ingestion:** Robust handling of document formats.
* **Multiple Embedding Strategies:** Comparative analysis of vector models.
* **Memory-Efficient:** Uses lazy loading to optimize resource usage.
* **LLM-Based Generation:** Integration with Groq (with fallback modes).
* **Scientific Evaluation:** Measures performance rather than guessing.
* **Auto-Selection:** automatically routes the query to the best configuration.

---

## Goal
The primary objective is to **determine which RAG configuration works best for a given document and query set — using measurement, not guesswork.**

---

## Tech Stack
* **Language:** Python
* **Embeddings:** Sentence Transformers
* **Vector Store:** ChromaDB
* **LLM:** Groq API (Optional/Configurable)
* **Interface:** Streamlit

![rag1](https://github.com/user-attachments/assets/a784f30b-a0b3-4d0e-bc97-2c84aa594244)
![rag2](https://github.com/user-attachments/assets/5f6e6a67-33a4-4976-acb4-590e43cf4306)




