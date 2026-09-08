# RAG Project – Energy Market Documents

A small Retrieval-Augmented Generation (RAG) prototype for semantic search in energy market and electricity-related documents.

## Project Goal

The goal of this project is to understand and implement the main components of a RAG system from scratch.

The current implementation focuses on the **retrieval part** of the RAG pipeline. An LLM generation component is intentionally not included yet.

## Architecture

```text
PDF Documents
      ↓
PDF Loader
      ↓
Document Chunking
      ↓
Text Embeddings
      ↓
ChromaDB
      ↓
Query Embedding
      ↓
Similarity Search
      ↓
Relevant Chunks
```

## Technologies

* Python
* LangChain
* Sentence Transformers
* ChromaDB
* Jupyter Notebook
* uv

## How it works

### 1. Document Loading

PDF documents are loaded using LangChain's `PyPDFLoader`.

Each PDF page is represented as a LangChain `Document` containing:

* page content
* source information
* page metadata

### 2. Chunking

The documents are split into smaller text chunks using `RecursiveCharacterTextSplitter`.

Current configuration:

```python
chunk_size = 1000
chunk_overlap = 200
```

The overlap helps preserve context between neighboring chunks.

### 3. Embeddings

Each text chunk is converted into a numerical vector using a Sentence Transformer model.

The current embedding model is:

```text
all-MiniLM-L6-v2
```

These vectors represent the semantic meaning of the text and can be compared to find similar content.

### 4. Vector Database

The embeddings, text and metadata are stored in a persistent ChromaDB collection.

The stored information includes:

* document text
* embedding
* source file
* page number
* additional metadata

### 5. Retrieval

When a user enters a question, the question is also converted into an embedding.

ChromaDB then searches for the most similar document chunks.

For example:

```python
rag_retriever.retrieve(
    "Erstelle einen Überblick über erneuerbare Energien",
    top_k=5
)
```

The retriever returns the most relevant chunks together with their metadata and similarity scores.

## Example

**Query:**

> Erstelle einen Überblick über erneuerbare Energien.

The system retrieves relevant passages from the energy market documents and returns information such as:

```text
Source: MonitoringberichtEnergie2024.pdf
Page: ...
Similarity score: ...
Content: ...
```

## Current Status

Implemented:

* [x] PDF document loading
* [x] Metadata extraction
* [x] Text chunking
* [x] Text embeddings
* [x] ChromaDB vector store
* [x] Semantic similarity search
* [x] Top-k document retrieval
* [x] Similarity score filtering

Planned:

* [ ] LLM integration
* [ ] Context generation
* [ ] Answer generation
* [ ] Source citations in generated answers
* [ ] Retrieval evaluation

## Project Structure

```text
rag_project/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── rag_experiments.ipynb
│
├── .gitignore
├── README.md
├── pyproject.toml
└── uv.lock
```

## Data

The project uses publicly available documents related to the German energy and electricity market.

The PDF files and generated ChromaDB vector store are not included in the repository.

## Note

This project is primarily a learning project to understand the architecture and implementation of Retrieval-Augmented Generation systems.
