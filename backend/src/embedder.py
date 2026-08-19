from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

