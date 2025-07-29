# Version 3 of the application, which replaces LangChain's CharacterTextSplitter with custom NLTK-based chunking
import os
import streamlit as st
import pdfplumber
import torch
import numpy as np
import nltk
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configuration
EMBEDDINGS_MODEL_NAME = "bert-base-uncased"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

class CustomEmbeddings:
    """
    Custom implementation of embeddings using PyTorch and BERT
    This replaces LangChain's HuggingFaceEmbeddings with our own PyTorch-based solution
    """
    def __init__(self, model_name=EMBEDDINGS_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Average pooling of token embeddings, weighted by attention mask
        This converts token-level embeddings to document-level embeddings
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_documents(self, texts):
        """
        Generate embeddings for a list of texts
        This method is required by LangChain's embedding interface
        """
        # Tokenize texts
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Apply mean pooling
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to numpy array
        return embeddings.cpu().numpy()
    
    def embed_query(self, text):
        """
        Generate embedding for a single text
        This method is required by LangChain's embedding interface
        """
        return self.embed_documents([text])[0]
    
    def __call__(self, texts):
        """
        Make the class callable - this is what LangChain expects
        This method allows the class to be used like: embeddings(texts)
        """
        if isinstance(texts, str):
            return self.embed_query(texts)
        else:
            return self.embed_documents(texts)

def nltk_sentence_chunker(text, max_chunk_size=500):
    """
    Custom text chunking using NLTK sentence tokenization
    This replaces LangChain's CharacterTextSplitter with our own NLTK-based solution
    
    Args:
        text (str): Input text to chunk
        max_chunk_size (int): Maximum size of each chunk in characters
    
    Returns:
        list: List of text chunks
    """
    # Split text into sentences using NLTK
    sentences = nltk.sent_tokenize(text)
    
    # Debug info
    st.info(f"NLTK found {len(sentences)} sentences in the document")
    
    # Group sentences into chunks
    chunks = []
    current_chunk = ""
    
    for i, sentence in enumerate(sentences):
        # If adding this sentence would exceed chunk size
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            # Add sentence to current chunk
            current_chunk += " " + sentence if current_chunk else sentence
    
    # the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    st.success(f"Created {len(chunks)} chunks using NLTK sentence tokenization")
    return chunks

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in .env file")

def extract_text_from_pdf(pdf_file):
    """
    Extract text from uploaded PDF file using pdfplumber.
    
    Args:
        pdf_file: Uploaded PDF file object
    Returns:
        str: Extracted text from PDF
    """
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None
    return text

def initialize_rag_components():
    """
    Initialize RAG pipeline components: embeddings, LLM, and prompt template.
    
    Returns:
        tuple: (embeddings, llm, prompt_template)
    """
    # Initialize custom embeddings (PyTorch-based)
    embeddings = CustomEmbeddings()
    
    # Initialize LLM (using GPT-3.5-turbo)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.1
    )
    
    # Create prompt template
    prompt_template = """Answer the following question based on the given context. If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context: {context}

Question: {question}

Answer:"""
    
    return embeddings, llm, PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

def create_retrieval_qa_chain(docs, embeddings, llm, prompt):
    """
    Create a retrieval QA chain from documents.
    
    Args:
        docs (str): Document text
        embeddings: Custom embeddings (PyTorch-based)
        llm: Language model
        prompt: Prompt template
    Returns:
        RetrievalQA: QA chain
    """
    # Use our custom NLTK chunking instead of LangChain's CharacterTextSplitter
    texts = nltk_sentence_chunker(docs, max_chunk_size=CHUNK_SIZE)
    
    # Create vector store using our custom embeddings
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': TOP_K_RESULTS}),
        chain_type_kwargs={
            'prompt': prompt,
            'verbose': True  # Enable verbose mode for debugging
        },
        return_source_documents=True
    )
    
    return qa_chain

def main():
    st.title("Talk to Your Docs 📚")
    st.write("Upload a PDF and ask questions about its content!")
    st.info("V3: Custom PyTorch embeddings + NLTK sentence chunking (replaces LangChain's CharacterTextSplitter)")
    
    # Initialize RAG components
    with st.spinner("Loading PyTorch model..."):
        embeddings, llm, prompt = initialize_rag_components()
    
    # File upload
    pdf_file = st.file_uploader("Upload your PDF", type=['pdf'])
    
    if pdf_file:
        # Extract text from PDF
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf(pdf_file)
            if text:
                st.success("PDF processed successfully!")
                
                # Create QA chain with custom NLTK chunking
                with st.spinner("Creating embeddings and vector store..."):
                    qa_chain = create_retrieval_qa_chain(text, embeddings, llm, prompt)
                
                # Question input
                question = st.text_input("Ask a question about your document:")
                
                if question:
                    with st.spinner("Searching for answer..."):
                        try:
                            # Get answer
                            result = qa_chain({"query": question})
                            
                            # Display answer
                            st.write("### Answer:")
                            if isinstance(result, dict) and "result" in result:
                                st.write(result["result"])
                            else:
                                st.write(str(result))  # Fallback for unexpected response format
                            
                            # Display source documents
                            if isinstance(result, dict) and "source_documents" in result:
                                with st.expander("View source chunks"):
                                    for i, doc in enumerate(result["source_documents"]):
                                        st.write(f"Chunk {i+1}:")
                                        st.write(doc.page_content)
                                        st.write("---")
                            
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")
                            st.error("Please try rephrasing your question or uploading a different document.")

if __name__ == "__main__":
    main()

# V3 Improvements:
# - Custom NLTK sentence chunking (replaces LangChain CharacterTextSplitter)
    # - Uses NLTK's sent_tokenize for sentence boundary detection
    # - Groups sentences into chunks based on character count
    # - Provides debugging information about chunking process

# TODO: Future V4+ Enhancements:
# - Direct FAISS integration (remove LangChain vectorstore)
# - Custom retrieval logic
# - HuggingFace LLM inference instead of OpenAI LLMs that cost money
# - Implement Arabic support using Arabic models
# - Advanced chunking with spaCy
 