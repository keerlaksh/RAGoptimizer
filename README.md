🧠 RAG Experimental Framework
Multi-Pipeline Retrieval-Augmented Generation (Research-Grade)
📌 What is this project?

This project is a research-grade Retrieval-Augmented Generation (RAG) system designed to experiment, analyze, and compare multiple RAG pipelines on the same documents and user queries.

Instead of building just another chatbot, this framework treats RAG as a scientific experiment:

Given the same documents and the same questions, which RAG design actually works best — and why?

🚀 Why this project exists (The real motivation)

Large Language Models (LLMs):

❌ Cannot read your private PDFs by default

❌ Hallucinate when asked about unknown content

❌ Have limited context windows

❌ Cannot cite document-grounded answers reliably

RAG fixes this by:

Retrieving relevant document chunks

Injecting them into the LLM prompt

Forcing answers to be grounded in real data

This project goes further by:

Comparing different embedding models

Comparing chunk sizes & strategies

Measuring hallucination vs groundedness

Quantifying retrieval → generation coupling

🧩 What does the system do?
Inputs

Documents: PDF / TXT

Queries: natural-language questions

Outputs

📊 Quantitative metrics (retrieval + generation)

🏆 Best-performing RAG pipeline

🧠 Final natural-language answer from the best pipeline

🔬 What makes this research-grade?

This framework explicitly measures:

1️⃣ Retrieval Quality

Hit Rate

MRR (Mean Reciprocal Rank)

MAP (Mean Average Precision)

nDCG@K

2️⃣ Generation Quality

BLEU

Readability (Flesch)

Fluency

Diversity & Novelty

3️⃣ End-to-End RAG Metrics

Groundedness

Hallucination Rate

Context Utilization

Answer Relevance

Response Coherence
