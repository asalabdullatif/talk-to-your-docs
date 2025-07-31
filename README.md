# Talk to Your Docs - RAG PDF Chatbot

A **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF documents and ask questions in natural language. Built with a focus on learning "under the hood" implementations of modern NLP/ML technologies.

## Project Overview

This project demonstrates the evolution of a RAG system from basic LangChain implementation to custom PyTorch-based components. The goal is to gain hands-on experience with:

- **PyTorch** and **Transformers** for custom embedding generation
- **NLTK** and **spaCy** for advanced NLP preprocessing
- **FAISS** for efficient vector similarity search
- **Arabic language support** with specialized models
- **OCR** for handling scanned documents

## Current Status

### Completed Versions

**V1: Foundation (LangChain + OpenAI)**
- Basic RAG pipeline with LangChain
- OpenAI GPT-3.5 for Q&A generation
- FAISS vector store for similarity search
- Streamlit UI for document upload and interaction
- PDF text extraction with `pdfplumber`

**V2: Custom PyTorch Embeddings**
- Replaced LangChain embeddings with custom PyTorch implementation
- HuggingFace Transformers (`bert-base-uncased`) for embedding generation
- Mean pooling for sentence-level representations
- L2 normalization for cosine similarity calculations

**V3: Custom NLTK Chunking**
- Replaced LangChain `CharacterTextSplitter` with NLTK sentence tokenization
- Sentence-level chunking with configurable overlap
- Interactive parameter testing (chunk size, overlap, top-k)
- Understanding NLP preprocessing fundamentals

### In Progress

**V4: Arabic Language Support**
- Arabic-optimized embeddings (`aubmindlab/bert-base-arabertv2`)
- Language detection and preprocessing
- spaCy for Arabic text processing
- OCR-based PDF extraction for Arabic documents

## Architecture

### RAG Pipeline Components

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Storage → Query → Retrieval → LLM → Answer
```

| Component | V1 (LangChain) | V2 (Custom) | V3 (Custom) | V4 (Planned) |
|-----------|----------------|-------------|-------------|---------------|
| **Text Extraction** | `pdfplumber` | `pdfplumber` | `pdfplumber` | OCR + `pdfplumber` |
| **Chunking** | `CharacterTextSplitter` | `CharacterTextSplitter` | **NLTK** | **spaCy** |
| **Embeddings** | `HuggingFaceEmbeddings` | **Custom PyTorch** | **Custom PyTorch** | **Arabic BERT** |
| **Vector Store** | FAISS | FAISS | FAISS | FAISS |
| **LLM for QA** | OpenAI GPT-3.5 | OpenAI GPT-3.5 | OpenAI GPT-3.5 | **HuggingFace-based Free LLM** |

### Key Technologies

- **Frontend:** Streamlit
- **Backend:** Python 3.10
- **ML Framework:** PyTorch 2.1.1 (CPU)
- **NLP Libraries:** NLTK, spaCy, Transformers
- **Vector Search:** FAISS
- **PDF Processing:** pdfplumber, PyMuPDF, OCR
- **Language Models:** OpenAI GPT-3.5, HuggingFace Transformers

## Installation

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd talk-to-your-docs
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
# Create .env file
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_TOKEN=your_huggingface_token
```

## Usage

### Running Different Versions

**V1 (LangChain + OpenAI):**
```bash
streamlit run app/app_v1.py
```

**V2 (Custom PyTorch Embeddings):**
```bash
streamlit run app/app_v2.py
```

**V3 (Custom NLTK Chunking):**
```bash
streamlit run app/app_v3.py
```

**V4 (Arabic Support) - Coming Soon:**
```bash
streamlit run app/app_v4.py
```