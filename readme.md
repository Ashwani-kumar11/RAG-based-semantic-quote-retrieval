# Task 2 – RAG-based Semantic Quote Retrieval & Structured QA

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to retrieve quotes from the [`Abirate/english_quotes`](https://huggingface.co/datasets/Abirate/english_quotes) dataset using semantic similarity. The system is built with a fine-tuned SentenceTransformer model, FAISS vector indexing, and is evaluated using RAGAS. It includes an interactive Streamlit application for end users to search quotes using natural language.

---

## Objective

> Given a query like: _"quotes about courage by women authors"_, retrieve and display the most relevant quotes, their authors, tags, and similarity scores in both JSON and formatted view.

---

## Components Overview

### 1. Data Preparation

- Dataset: [`Abirate/english_quotes`](https://huggingface.co/datasets/Abirate/english_quotes)
- Fields: `quote`, `author`, `tags`
- Preprocessing:
  - Lowercased text
  - Removed empty/missing fields
  - Saved cleaned data to `cleaned_quotes.json`

### 2. Model Fine-Tuning

- Base model: `all-MiniLM-L6-v2`
- Fine-tuned using `MultipleNegativesRankingLoss`
- Created synthetic training pairs:
  - Query: `"quotes about {tag} by {author}"`
  - Positive: actual quote

### 3. Vector Indexing (FAISS)

- Quotes embedded using the fine-tuned model
- Indexed using `faiss.IndexFlatL2` for fast similarity search

### 4. Retrieval Function

- Given a natural language query, the system:
  - Encodes the query
  - Retrieves top-k matching quotes from FAISS
  - Returns results with similarity scores

### 5. RAG Evaluation with RAGAS

- Metrics used:
  - `faithfulness`
  - `answer_relevancy`
  - `context_precision`
  - `context_recall`
- Sample results generated and evaluated
- Requires OpenAI API key (set via `os.environ["OPENAI_API_KEY"]`)

### 6. Streamlit Application

- Allows user to input a query
- Shows retrieved quotes in:
  - JSON structure
  - Clean formatted view
- Also displays similarity scores

---

## Running the App

### 🔹 1. Install Requirements

```bash
pip install -r requirements.txt
