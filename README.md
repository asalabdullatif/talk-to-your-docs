# Talk to Your Docs

A Streamlit application that allows you to upload PDF documents and ask questions about their content. The app uses LangChain, HuggingFace models, and FAISS for efficient document question-answering.

## Features

- PDF document upload and text extraction
- Question-answering using HuggingFace models
- Efficient document chunking and retrieval
- User-friendly Streamlit interface

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your HuggingFace API token:
```
HUGGINGFACE_API_TOKEN=your_token_here
```

4. Run the application:
```bash
streamlit run app.py
```

## Requirements

- Python 3.10+
- See requirements.txt for full list of dependencies