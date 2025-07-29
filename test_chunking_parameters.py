import nltk
import streamlit as st

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def test_chunking_parameters(text, chunk_size, chunk_overlap=50):
    """
    Test different chunking parameters and show results
    """
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for i, sentence in enumerate(sentences):
        if len(current_chunk) + len(sentence) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Start new chunk with sentence-level overlap
            if chunk_overlap > 0 and chunks:
                # Get the last chunk and find sentences that fit within overlap
                last_chunk = chunks[-1]
                last_chunk_sentences = nltk.sent_tokenize(last_chunk)
                
                # Calculate how much overlap we can actually use
                # We want to include sentences from the end of the previous chunk
                overlap_sentences = []
                total_overlap_chars = 0
                
                # Start from the end and work backwards
                for sent in reversed(last_chunk_sentences):
                    sent_length = len(sent)
                    # Check if adding this sentence would exceed our overlap limit
                    if total_overlap_chars + sent_length <= chunk_overlap:
                        overlap_sentences.insert(0, sent)  # Insert at beginning to maintain order
                        total_overlap_chars += sent_length
                    else:
                        # This sentence is too long for our overlap, stop here
                        break
                
                # Create the new chunk with overlap
                if overlap_sentences:
                    # Join the overlapping sentences and add the new sentence
                    overlap_text = " ".join(overlap_sentences)
                    current_chunk = overlap_text + " " + sentence
                else:
                    # No overlap possible, start fresh
                    current_chunk = sentence
            else:
                # No overlap requested or no previous chunks
                current_chunk = sentence
        else:
            # Add sentence to current chunk
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def analyze_chunks(chunks):
    """
    Analyze chunk statistics
    """
    if not chunks:
        return {}
    
    chunk_lengths = [len(chunk) for chunk in chunks]
    
    return {
        'num_chunks': len(chunks),
        'avg_length': sum(chunk_lengths) / len(chunk_lengths),
        'min_length': min(chunk_lengths),
        'max_length': max(chunk_lengths),
        'total_chars': sum(chunk_lengths)
    }

# Test with different parameters
test_text = """
Machine learning is a subset of artificial intelligence that focuses on developing algorithms and statistical models that enable computers to learn and make predictions or decisions without being explicitly programmed. The field has seen tremendous growth in recent years, with applications ranging from image recognition and natural language processing to autonomous vehicles and medical diagnosis.

Deep learning, a subset of machine learning, uses neural networks with multiple layers to model and understand complex patterns in data. These neural networks are inspired by the human brain's structure and function, consisting of interconnected nodes that process information through multiple layers of abstraction.

The success of machine learning depends heavily on the quality and quantity of training data. More data generally leads to better model performance, but the data must also be relevant, clean, and properly labeled. Data preprocessing, feature engineering, and model selection are crucial steps in the machine learning pipeline.

Supervised learning involves training models on labeled data, where the correct output is known for each input. Common supervised learning tasks include classification (predicting categories) and regression (predicting continuous values). Unsupervised learning, on the other hand, works with unlabeled data to discover hidden patterns and structures.
"""

# Interactive testing
st.write("### Interactive Testing")
col1, col2 = st.columns(2)

with col1:
    user_chunk_size = st.slider("Chunk Size", 100, 1000, 500, 50)
    user_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)

with col2:
    st.write(f"**Overlap %:** {(user_overlap/user_chunk_size)*100:.1f}%")
    st.write("**Effects:**")
    if user_overlap == 0:
        st.write("- No overlap: Clean chunks")
    elif user_overlap < user_chunk_size * 0.2:
        st.write("- Small overlap: Minimal redundancy")
    elif user_overlap < user_chunk_size * 0.4:
        st.write("- Medium overlap: Good context preservation")
    else:
        st.write("- Large overlap: High redundancy")

user_text = st.text_area("Enter your own text to test:", value=test_text, height=200)

if st.button("Test Chunking"):
    user_chunks = test_chunking_parameters(user_text, user_chunk_size, user_overlap)
    user_stats = analyze_chunks(user_chunks)
    
    st.write(f"**Results for chunk size {user_chunk_size} with {user_overlap} overlap:**")
    st.write(f"- Created {user_stats['num_chunks']} chunks")
    st.write(f"- Average length: {user_stats['avg_length']:.1f} characters")
    st.write(f"- Overlap percentage: {(user_overlap/user_chunk_size)*100:.1f}%")
    
    st.write("**All chunks:**")
    for i, chunk in enumerate(user_chunks):
        with st.expander(f"Chunk {i+1} ({len(chunk)} chars)"):
            st.write(chunk)
            
            # Show overlap info if there's overlap
            if user_overlap > 0 and i > 0:
                # Check if this chunk starts with sentences from previous chunk
                prev_chunk_sentences = nltk.sent_tokenize(user_chunks[i-1])
                current_chunk_sentences = nltk.sent_tokenize(chunk)
                
                # Find overlapping sentences
                overlap_found = False
                for prev_sent in prev_chunk_sentences:
                    if prev_sent in chunk:
                        st.info(f"🔗 Overlap: '{prev_sent[:50]}...' appears in both chunks")
                        overlap_found = True
                        break
                
                if not overlap_found:
                    st.warning("⚠️ No sentence overlap detected - might be too small of an overlap") 