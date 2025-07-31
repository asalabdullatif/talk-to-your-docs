# Test script for Arabic language support
import streamlit as st
import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
from langdetect import detect, LangDetectException

st.title("Arabic Language Support Test")

# Sample texts
english_text = "This is a sample English text for testing language detection and embeddings."
arabic_text = "هذا نص عربي للاختبار. نحن نختبر كشف اللغة العربية ومعالجة النصوص."

def detect_language(text):
    """Detect if text is Arabic or English"""
    try:
        # Clean text for language detection
        clean_text = re.sub(r'[^\w\s]', '', text[:1000])  # Use first 1000 chars
        if not clean_text.strip():
            return "english"  # Default to English
        
        lang = detect(clean_text)
        return "arabic" if lang == "ar" else "english"
    except LangDetectException:
        return "english"  # Default to English

def preprocess_arabic_text(text):
    """Preprocess Arabic text for better processing"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize Arabic characters
    text = text.replace('ي', 'ی')  # Normalize ya
    text = text.replace('ك', 'ک')  # Normalize kaf
    
    return text.strip()

def test_language_detection():
    """Test language detection functionality"""
    st.header("Language Detection Test")
    
    # Test English
    eng_lang = detect_language(english_text)
    st.write(f"**English text:** {english_text}")
    st.write(f"**Detected language:** {eng_lang}")
    st.write("---")
    
    # Test Arabic
    ar_lang = detect_language(arabic_text)
    st.write(f"**Arabic text:** {arabic_text}")
    st.write(f"**Detected language:** {ar_lang}")
    st.write("---")
    
    # Test user input
    user_text = st.text_input("Enter text to test language detection:")
    if user_text:
        detected = detect_language(user_text)
        st.write(f"**Detected language:** {detected}")

def test_arabic_preprocessing():
    """Test Arabic text preprocessing"""
    st.header("Arabic Text Preprocessing Test")
    
    # Original Arabic text
    st.write("**Original Arabic text:**")
    st.write(arabic_text)
    
    # Preprocessed text
    processed = preprocess_arabic_text(arabic_text)
    st.write("**Preprocessed Arabic text:**")
    st.write(processed)
    
    # Test with user input
    user_arabic = st.text_input("Enter Arabic text to preprocess:")
    if user_arabic:
        processed_user = preprocess_arabic_text(user_arabic)
        st.write("**Preprocessed:**")
        st.write(processed_user)

def test_embeddings():
    """Test embedding generation for both languages"""
    st.header("Embedding Generation Test")
    
    # Model names
    arabic_model = "aubmindlab/bert-base-arabertv2"
    english_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    try:
        with st.spinner("Loading Arabic BERT model..."):
            arabic_tokenizer = AutoTokenizer.from_pretrained(arabic_model)
            arabic_model_obj = AutoModel.from_pretrained(arabic_model)
            arabic_model_obj.eval()
        
        with st.spinner("Loading English BERT model..."):
            english_tokenizer = AutoTokenizer.from_pretrained(english_model)
            english_model_obj = AutoModel.from_pretrained(english_model)
            english_model_obj.eval()
        
        st.success("Models loaded successfully!")
        
        # Test Arabic embeddings
        st.write("**Testing Arabic embeddings:**")
        arabic_encoded = arabic_tokenizer(arabic_text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            arabic_output = arabic_model_obj(**arabic_encoded)
        
        # Mean pooling for Arabic
        arabic_embeddings = arabic_output.last_hidden_state
        arabic_attention_mask = arabic_encoded['attention_mask']
        arabic_mask_expanded = arabic_attention_mask.unsqueeze(-1).expand(arabic_embeddings.size()).float()
        arabic_pooled = torch.sum(arabic_embeddings * arabic_mask_expanded, 1) / torch.clamp(arabic_mask_expanded.sum(1), min=1e-9)
        arabic_normalized = torch.nn.functional.normalize(arabic_pooled, p=2, dim=1)
        
        st.write(f"Arabic embedding shape: {arabic_normalized.shape}")
        st.write(f"Arabic embedding sample: {arabic_normalized[0][:5].tolist()}")
        
        # Test English embeddings
        st.write("**Testing English embeddings:**")
        english_encoded = english_tokenizer(english_text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            english_output = english_model_obj(**english_encoded)
        
        # Mean pooling for English
        english_embeddings = english_output.last_hidden_state
        english_attention_mask = english_encoded['attention_mask']
        english_mask_expanded = english_attention_mask.unsqueeze(-1).expand(english_embeddings.size()).float()
        english_pooled = torch.sum(english_embeddings * english_mask_expanded, 1) / torch.clamp(english_mask_expanded.sum(1), min=1e-9)
        english_normalized = torch.nn.functional.normalize(english_pooled, p=2, dim=1)
        
        st.write(f"English embedding shape: {english_normalized.shape}")
        st.write(f"English embedding sample: {english_normalized[0][:5].tolist()}")
        
        # Test similarity (only within same language)
        st.write("**Testing within-language similarity:**")
        
        # Test Arabic similarity with itself
        arabic_similarity = torch.cosine_similarity(arabic_normalized, arabic_normalized, dim=1)
        st.write(f"Arabic self-similarity: {arabic_similarity.item():.4f}")
        
        # Test English similarity with itself  
        english_similarity = torch.cosine_similarity(english_normalized, english_normalized, dim=1)
        st.write(f"English self-similarity: {english_similarity.item():.4f}")
        
        st.info("ℹNote: Cross-language similarity not calculated due to different embedding dimensions (768 vs 384)")
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Make sure you have the required models installed.")

def main():
    st.info("This script tests Arabic language support functionality for V4.")
    
    # Test sections
    test_language_detection()
    st.write("---")
    test_arabic_preprocessing()
    st.write("---")
    test_embeddings()

if __name__ == "__main__":
    main() 