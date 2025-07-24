# Talk to Your Docs - RAG PDF Chatbot

## Project Overview
A demonstration project implementing a PDF chatbot using Retrieval-Augmented Generation (RAG). The goal is to create a simple demo as part of learning and demonstrating NLP/RAG capabilities.

## Development Plan

### Version 1 (Current) - LangChain-based RAG
- Simple Streamlit interface for PDF upload and Q&A
- Uses LangChain for RAG pipeline
- Free HuggingFace models for embeddings and LLM
- Basic functionality without bells and whistles

### Future Versions - Custom RAG Implementation
- Remove LangChain dependency
- Implement custom RAG using PyTorch
- Direct use of Transformers library
- Arabic language support


## Learning Objectives
1. RAG Architecture and Implementation
   - Document processing and chunking
   - Vector embeddings and similarity search
   - Context retrieval and prompt engineering

2. Key Technologies
   - LangChain (v1)
   - PyTorch (v2)
   - Transformers library
   - FAISS vector store
   - Streamlit for UI

## Current Status
- Setting up initial scaffold with LangChain
- Basic PDF processing and Q&A functionality

## Next Steps
- Complete v1 implementation
- Test with various PDF types
- Plan v2 custom implementation, and which ones to prioriize and tackle

## Notes & Learnings

## Resources
- HuggingFace Models:
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2
  - LLM: google/flan-t5-base