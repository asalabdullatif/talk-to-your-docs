# Talk to Your Docs - RAG PDF Chatbot

## Project Overview

A **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF documents and ask questions in natural language. The system progressively evolved from a basic LangChain implementation to custom PyTorch-based components, demonstrating hands-on experience with modern NLP/ML technologies.

**Key Goals:**
- Learn "under the hood" implementations of RAG components
- Gain practical experience with PyTorch, Transformers, and NLP libraries
- Build a system that works with both English and Arabic documents
- Demonstrate advanced RAG capabilities beyond "plain vanilla" implementations

## Planning & Strategy

### Initial Component Breakdown

The project was strategically broken down into **4 core components**, each with multiple implementation options ranging from simple to advanced:

| Component | Purpose | Simple Option | Advanced Option |
|-----------|---------|---------------|-----------------|
| **Text Ingestion** | Extract clean text from PDFs | `pdfplumber`, `PyMuPDF` | OCR with `pytesseract` |
| **Chunking + Embedding** | Break text into parts and vectorize | LangChain + OpenAI | Custom PyTorch + BERT |
| **Retrieval** | Find relevant chunks to questions | LangChain retriever | Manual cosine similarity + FAISS |
| **Answer Generation** | Generate answers using LLM | OpenAI GPT API | HuggingFace LLMs |

### Version Strategy & Learning Roadmap

**V1: Foundation (LangChain + OpenAI)**
- Basic RAG pipeline with LangChain
- OpenAI GPT-3.5 for Q&A
- FAISS for vector storage
- Streamlit UI

**V2: Custom Embeddings (PyTorch)**
- Replace LangChain embeddings with custom PyTorch implementation
- Use HuggingFace Transformers (`bert-base-uncased`)
- Implement mean pooling for sentence embeddings

**V3: Custom Chunking (NLTK)**
- Replace LangChain `CharacterTextSplitter` with NLTK sentence tokenization
- Implement sentence-level chunking with overlap
- Learn NLP preprocessing fundamentals
- Understand chunking parameters (size, overlap, top-k)

**V4: Arabic Language Support (Planned)**
- Arabic-optimized embeddings (`aubmindlab/bert-base-arabertv2`)
- Language detection and preprocessing
- spaCy for Arabic text processing
- OCR-based PDF extraction for Arabic documents

### Priority Matrix for Job Requirements

Based on the target job description, components were prioritized by **importance** and **learning value**:

| Task | Priority | Impact | Difficulty | Core Tools |
|------|----------|--------|------------|------------|
| Replace LangChain | Must | High | Medium | PyTorch, Transformers |
| Arabic Embeddings | Must | High | Medium | CAMeL, AraBERT |
| Use HF LLM | Must | Medium-High | Medium | Transformers |
| Fine-Tune Embedder/Reranker | Optional | Very High | High | PyTorch |
| NLP Preprocessing | Recommended | Medium | Low | spaCy, NLTK |
| Model Serving | Optional | Medium | High | FastAPI |


## 🧠 Key Learnings

### Project Planning & Development Strategy

- **MVP-First Approach**: Always start with a POC/MVP as V1, then progressively improve each component
- **Component Isolation**: Replace one component at a time to understand its role and dependencies
- **Learning-Focused Design**: Choose tools that expose underlying mechanisms (NLTK over spaCy for learning)
- **Version Control Strategy**: Maintain separate versions to demonstrate evolution and learning progression

### Technical Architecture Insights

**RAG Pipeline Understanding:**
- **Why chunking?** LLMs have token limits, so we only send relevant chunks to answer questions
- **Retrieval mechanism:** FAISS stores vectors and finds similar ones using cosine similarity
- **Context injection:** Top-K most similar chunks are fed to LLM as context for answer generation

**LangChain-Specific Learnings:**
- Prompt templates initialize variables (`context`, `question`) that get filled during chain execution
- `context` gets filled with top-K retrieved chunks
- `question` gets filled when calling `qa_chain({"query": question})`
- Embedding functions must be proper Embeddings objects, not callable functions

### Deep Learning & NLP Fundamentals

**Transformers Architecture:**
- **Three types:** Encoder-Decoder (BART), Decoder (GPT), Encoder (BERT)
- **Key components:** Word embeddings, positional encoding, self-attention, residual connections
- **Self-attention:** Enables understanding relationships between all words in a sentence
- **Positional encoding:** Uses sine/cosine functions to encode word positions

**PyTorch Fundamentals:**
- **Tensors vs NumPy:** GPU acceleration, automatic differentiation, better for deep learning
- **Autograd:** Automatic gradient calculation for backpropagation
- **Neural Network structure:** `nn.Module` base class, `__init__` for layers, `forward()` for data flow
- **Training workflow:** Loss calculation → `backward()` → `optimizer.step()` → `zero_grad()`

**Embedding Generation:**
- **Mean pooling:** Average token embeddings to get sentence-level representations
- **Normalization:** L2 normalization for cosine similarity calculations
- **Model loading:** HuggingFace `AutoTokenizer` and `AutoModel` for easy model switching

### Tool Selection & Trade-offs

**NLTK vs spaCy:**
- **NLTK:** Slower, more manual, research-focused, exposes lower-level building blocks
- **spaCy:** Faster, more advanced, production-ready, abstracts away complexity
- **Decision:** NLTK for learning (V3), spaCy for production features (V4)

**FAISS vs Cosine Similarity:**
- **FAISS:** Optimized vector similarity search, handles large-scale operations
- **Cosine similarity:** Simple mathematical operation, good for understanding
- **Relationship:** FAISS uses cosine similarity internally but optimizes the search process

**Chunking Parameters:**
- **CHUNK_SIZE = 500:** Middle-ground (200-1000 chars), ~100-150 tokens
- **CHUNK_OVERLAP = 50:** Prevents context loss at boundaries
- **TOP_K_RESULTS = 3:** Balances precision vs context richness

### API Costs & Performance

**OpenAI vs HuggingFace: Embeddings & QA**

- **OpenAI LLMs and Embeddings:** Accessed via API calls, which can incur usage costs and are subject to rate limits and network latency. They offer strong performance and easy integration, but require sending data to external servers.
- **HuggingFace (Local) Models:** Run locally, allowing for full control over data privacy and no per-request costs. They require more setup and sufficient hardware (ideally with GPU acceleration) for optimal performance, but enable faster inference and customization.

**Performance Considerations:**
- Local inference (HuggingFace) can be faster with the right hardware, while API-based solutions (OpenAI) may introduce latency.
- Model size, architecture, and hardware availability all impact speed and accuracy.
- Choosing between OpenAI and HuggingFace involves trade-offs between cost, privacy, scalability, and ease of use.

## Next Steps & Future Development

### Immediate Priorities (V4)

1. **Arabic Language Support**
   - Implement language detection
   - Arabic text preprocessing (normalization, diacritic removal)
   - Arabic-optimized embeddings (`aubmindlab/bert-base-arabertv2`)
   - OCR-based PDF extraction for Arabic documents

2. **Advanced RAG Features**
   - Custom re-ranker for improved retrieval
   - Multi-modal support (images + text)
   - Conversation memory and context management

### Long-term Roadmap

1. **Model Fine-tuning**
   - Domain-specific embedding fine-tuning
   - Custom re-ranker training
   - End-to-end RAG pipeline optimization

2. **Production Features**
   - FastAPI backend for model serving
   - Database integration for persistent storage
   - Authentication and user management
   - API rate limiting and monitoring

3. **Advanced NLP**
   - Named Entity Recognition (NER)
   - Document classification
   - Multi-language support expansion
   - Semantic search improvements

### Learning Objectives

**PyTorch Mastery:**
- Neural network architecture design
- Custom loss functions and training loops
- Model optimization and deployment
- GPU acceleration and distributed training

**NLP Deep Dive:**
- Advanced tokenization techniques
- Attention mechanisms and transformers
- Pre-training and fine-tuning strategies
- Evaluation metrics and benchmarking

**System Design:**
- Scalable architecture patterns
- Performance optimization
- Monitoring and observability
- Security and privacy considerations

## Resources & References

### Key Learning Materials
- **Transformers:** Jay Alammar's "The Illustrated Transformer"
- **PyTorch:** Official 60-minute blitz tutorial
- **NLP:** NLTK documentation and spaCy tutorials
- **RAG:** LangChain documentation and community examples
- **General:** Youtube channels such as Statquest & 3blue1brown 

### Technical References
- **FAISS:** Facebook AI Similarity Search documentation
- **HuggingFace:** Transformers library tutorials
- **Arabic NLP:** CAMeL Lab resources and AraBERT models
- **OCR:** Tesseract documentation and Arabic language packs