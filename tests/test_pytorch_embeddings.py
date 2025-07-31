import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def test_pytorch_embeddings():
    """
    Minimal test script to demonstrate PyTorch embeddings
    """
    print("=== PyTorch Embeddings Test Script ===\n")
    
    # 1. Load model and tokenizer
    print("1. Loading BERT model and tokenizer...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # 2. Test texts
    test_texts = [
        "Hello world, this is a test sentence.",
        "PyTorch is great for deep learning.",
        "Embeddings convert text to numbers."
    ]
    
    print(f"\n2. Test texts:")
    for i, text in enumerate(test_texts):
        print(f"   {i+1}. {text}")
    
    # 3. Tokenize texts
    print(f"\n3. Tokenizing texts...")
    encoded_input = tokenizer(
        test_texts,
        padding=True,           # Pad shorter sequences
        truncation=True,        # Truncate longer sequences
        max_length=512,         # Maximum sequence length
        return_tensors='pt'     # Return PyTorch tensors
    )
    
    print(f"   Input shape: {encoded_input['input_ids'].shape}")
    print(f"   Attention mask shape: {encoded_input['attention_mask'].shape}")
    
    # 4. Generate embeddings
    print(f"\n4. Generating embeddings...")
    with torch.no_grad():  # Don't compute gradients (faster inference)
        model_output = model(**encoded_input)
    
    # model_output contains:
    # - last_hidden_state: (batch_size, sequence_length, hidden_size)
    # - pooler_output: (batch_size, hidden_size) - only for some models
    print(f"   Model output keys: {model_output.keys()}")
    print(f"   Last hidden state shape: {model_output.last_hidden_state.shape}")
    
    # 5. Mean pooling (average the token embeddings)
    print(f"\n5. Applying mean pooling...")
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_size)
    attention_mask = encoded_input['attention_mask']    # (batch_size, seq_len)
    
    # Expand attention mask to match embedding dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Weighted average: sum embeddings * mask, then divide by sum of mask
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    print(f"   Final embeddings shape: {embeddings.shape}")
    print(f"   Each text is now represented by {embeddings.shape[1]} numbers")
    
    # 6. Normalize embeddings (for cosine similarity)
    print(f"\n6. Normalizing embeddings...")
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # 7. Convert to numpy and show some stats
    embeddings_np = normalized_embeddings.cpu().numpy()
    print(f"\n7. Embedding statistics:")
    print(f"   Mean: {embeddings_np.mean():.4f}")
    print(f"   Std: {embeddings_np.std():.4f}")
    print(f"   Min: {embeddings_np.min():.4f}")
    print(f"   Max: {embeddings_np.max():.4f}")
    
    # 8. Calculate similarities between texts
    print(f"\n8. Calculating similarities between texts:")
    similarities = np.dot(embeddings_np, embeddings_np.T)
    for i in range(len(test_texts)):
        for j in range(i+1, len(test_texts)):
            sim = similarities[i, j]
            print(f"   Similarity between text {i+1} and {j+1}: {sim:.4f}")
    
    print(f"\n=== Test completed successfully! ===")
    return embeddings_np

if __name__ == "__main__":
    # Run the test
    embeddings = test_pytorch_embeddings()
    
    print(f"\nYou can now use these embeddings for:")
    print(f"- Vector similarity search")
    print(f"- Document clustering")
    print(f"- Semantic search")
    print(f"- And more!") 