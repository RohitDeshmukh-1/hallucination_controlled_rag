from sentence_transformers import SentenceTransformer
import sys

try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    print("Model loaded successfully.")
    embeddings = model.encode(["Hello world"])
    print(f"Embeddings shape: {embeddings.shape}")
except Exception as e:
    print(f"Error loading/using model: {e}")
    sys.exit(1)
