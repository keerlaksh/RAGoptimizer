🧠 RAG Experimental Framework

This project is a research-grade Retrieval-Augmented Generation (RAG) system designed to compare multiple RAG configurations and identify which setup performs best for a given set of documents and queries.

❓ Why RAG?

Large Language Models (LLMs) do not truly read PDFs and do not remember uploaded files by default.
They may hallucinate or miss important details.

RAG solves this by:

Extracting text from documents (PDF/TXT)

Splitting text into meaningful chunks

Converting chunks into embeddings (vectors)

Retrieving the most relevant chunks for a question

Feeding only that context to the LLM for a grounded answer

🔬 What Makes This Project Different?

This is not just a chatbot.

It:

Runs multiple RAG pipelines (different embeddings & chunk sizes)

Answers the same question using each pipeline

Evaluates performance (groundedness, hallucination, context usage)

Automatically selects the best-performing pipeline

🚀 Key Features

📄 PDF upload & processing

🧩 Multiple chunking + embedding strategies

🤖 LLM-based answers using retrieved context

📊 Scientific evaluation & comparison

🏆 Best-pipeline selection

🎯 Goal

“Given the same documents and questions, which RAG configuration produces the most accurate, grounded answers?”

This framework helps you prove, not guess, what works best.
