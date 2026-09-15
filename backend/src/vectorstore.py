import json
import os
import shutil
from sentence_transformers import CrossEncoder
import warnings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from backend.src.embedder import get_embedder
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from backend.src.profile_text import profile_to_text
# Hide the LibreSSL message so our verification console looks clean
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# FORCED PATHS: We explicitly set paths to prevent OS environment paths from mismatching
from backend.core.config import CHROMA_DIR, PROFILES_PATH, COLLECTION_NAME
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def final_production_search(query, top_k=3):
    """
    Stage 1: Hybrid Search (Chroma + BM25) ensures high recall.
    Stage 2: Cross-Encoder Reranking ensures hyper-precise precision.
    """
    # ---- STAGE 1: HYBRID RETRIEVAL (Ensures we don't miss exact keywords) ----
    chroma_db = Chroma(
        collection_name="people_collection_v1_2",
        persist_directory=CHROMA_DIR,
        embedding_function=get_embedder(),
    )
    # Pull 10 total candidates from vector space
    vector_retriever = chroma_db.as_retriever(search_kwargs={"k": 10})

    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    
    documents = [
        Document(page_content=profile_to_text(p), metadata={"name": p.get("name"), "source": p.get("source")})
        for p in profiles
    ]
    
    # Pull 10 total candidates from keyword space
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10

    # Ensemble them 50/50 to catch a wide pool of unique candidate possibilities
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    # Run Stage 1 retrieval lookup
    hybrid_candidates = ensemble_retriever.invoke(query)

    if not hybrid_candidates:
        return []


    
    # Pair the query with each hybrid candidate text block
    pairs = [[query, doc.page_content] for doc in hybrid_candidates]
    scores = reranker.predict(pairs)

    # Bind scores back to original documents
    scored_docs = []
    for doc, score in zip(hybrid_candidates, scores):
        scored_docs.append((doc, float(score)))

    # Sort descending by high-relevance score matching weights
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return scored_docs[:top_k]


if __name__ == "__main__":
    # 1. Force remove any local 'fresh_chroma_db' folder to wipe previous memory
    if os.path.exists(CHROMA_DIR):
        print(f"Wiping out previous cache directory at: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    # 2. Extract profile JSON data
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    texts = []
    metadatas = []
    for record in profiles:
        texts.append(profile_to_text(record))
        metadatas.append(
            {
                "name": record.get("name") or "Unknown",
                "source": record.get("source") or "Unknown",
            }
        )

    # 3. Create a brand-new database instance in the isolated path
    db = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=get_embedder(),
        collection_name="people_collection_v1_2",  # Renamed collection to completely invalidate any lingering memory cache
        persist_directory=CHROMA_DIR,
    )


    print("VERIFICATION CHECK")
    print(f"Target Storage Location : {CHROMA_DIR}")
    print(f"Total Profiles Written : {db._collection.count()}")

