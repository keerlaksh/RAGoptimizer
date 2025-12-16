RAG Experimental Framework

A research-grade Retrieval-Augmented Generation (RAG) system that compares multiple RAG pipelines and identifies the best configuration for a given document and query set.

Overview

Large Language Models (LLMs) cannot reliably answer questions from long documents on their own.
They may hallucinate, miss details, or rely on prior knowledge.

This project solves that problem using Retrieval-Augmented Generation (RAG) and treats RAG design as a controlled experiment.

Why RAG?

RAG improves LLM answers by:

Extracting text from documents (PDF / TXT)

Splitting text into chunks

Converting chunks into vector embeddings

Retrieving the most relevant chunks per query

Generating answers strictly from retrieved context

This ensures answers are grounded, accurate, and verifiable.

What This Project Does

Builds multiple RAG pipelines using different:

Embedding models

Chunk sizes

Answers the same query using each pipeline

Evaluates each pipeline scientifically

Selects the best-performing pipeline

Returns:

Final answer

Evaluation metrics

Pipeline comparison report

Key Features

PDF document ingestion

Multiple embedding strategies

Lazy-loaded, memory-efficient indexing

LLM-based answer generation (Groq / fallback mode)

Quantitative RAG evaluation

Best-pipeline selection

Goal

Determine which RAG configuration works best for a given document and query set — using evidence, not intuition.

Tech Stack

Python

Sentence Transformers

ChromaDB

Groq LLM API (optional)

Streamlit
