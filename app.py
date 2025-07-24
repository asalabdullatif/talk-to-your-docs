import os
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configuration
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

# Check for HuggingFace API token
if not os.getenv("HUGGINGFACE_API_TOKEN"):
    raise ValueError("Please set HUGGINGFACE_API_TOKEN in .env file")

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
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
    # Initialize LLM
    llm = HuggingFaceHub(
        repo_id=LLM_MODEL_NAME,
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
        model_kwargs={'temperature': 0.5, 'max_length': 512}
    )
    
    # Create prompt template
    # TODO: Experiment with different prompt templates for better results
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
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
        embeddings: HuggingFace embeddings
        llm: Language model
        prompt: Prompt template
    Returns:
        RetrievalQA: QA chain
    """
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    texts = text_splitter.split_text(docs)
    
    # Create vector store
    # TODO: Add persistence to save embeddings for future use
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': TOP_K_RESULTS}),
        chain_type_kwargs={'prompt': prompt}
    )
    
    return qa_chain

def main():
    st.title("Talk to Your Data 📚")
    st.write("Upload a PDF and ask questions about its content!")
    
    # Initialize RAG components
    embeddings, llm, prompt = initialize_rag_components()
    
    # File upload
    pdf_file = st.file_uploader("Upload your PDF", type=['pdf'])
    
    if pdf_file:
        # Extract text from PDF
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf(pdf_file)
            if text:
                st.success("PDF processed successfully!")
                
                # Create QA chain
                qa_chain = create_retrieval_qa_chain(text, embeddings, llm, prompt)
                
                # Question input
                question = st.text_input("Ask a question about your document:")
                
                if question:
                    with st.spinner("Searching for answer..."):
                        try:
                            # Get answer
                            response = qa_chain.run(question)
                            
                            # Display answer
                            st.write("### Answer:")
                            st.write(response)
                            
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()

# TODO: Future Enhancements
# 1. Add support for multiple PDF uploads
# 2. Implement chat history and conversation memory
# 3. Add FAISS index persistence to disk
# 4. Make embedding and LLM models configurable via UI (could be deprioritized)
# 5. Add error handling and retry logic for API calls (could be deprioritized)
# 6. Implement progress bars for document processing (could be deprioritized)
# 7. Add option to view retrieved context chunks (could be deprioritized)
# 8. Implement Arabic support using Arabic models
# 9. Add document preprocessing options (e.g., remove headers/footers) (could be deprioritized)
# 10. Add example PDFs and questions for demo purposes 