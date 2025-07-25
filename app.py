import os
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configuration
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

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
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    
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
        length_function=len,
        separator="\n"  # Split on newlines for better context preservation
    )
    texts = text_splitter.split_text(docs)
    
    # Create vector store
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

# TODO: Future Enhancements
# 1. Add support for multiple PDF uploads
# 2. Implement chat history and conversation memory
# 3. Add FAISS index persistence to disk
# 4. Implement Arabic support using Arabic models (important)
# 5. Add example PDFs and questions for demo purposes 
# 6. Change from Openai LLM to a HF free model (important)