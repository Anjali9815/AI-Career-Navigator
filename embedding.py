# See what an embedding actually looks like

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def get_embeddings_model():
    print(f"\n🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True}  # needed for cosine similarity
    )
    print(f"✅ Embedding model loaded (384 dimensions per chunk)")
    return embeddings

embeddings_model = get_embeddings_model()

test_text = "Python machine learning experience"
vector = embeddings_model.embed_query(test_text)

print(f"Text: '{test_text}'")
print(f"Vector dimensions: {len(vector)}")
print(f"First 10 numbers: {[round(x, 4) for x in vector[:10]]}")
print(f"Min value: {min(vector):.4f}")
print(f"Max value: {max(vector):.4f}")

# Compare two similar sentences
v1 = embeddings_model.embed_query("Python developer with ML skills")
v2 = embeddings_model.embed_query("Machine learning engineer using Python")
v3 = embeddings_model.embed_query("I enjoy cooking pasta")

import numpy as np
def cosine_sim(a, b):
    return round(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 4)

print(f"\nSimilarity (Python dev vs ML engineer) : {cosine_sim(v1, v2)}")
print(f"Similarity (Python dev vs cooking pasta): {cosine_sim(v1, v3)}")