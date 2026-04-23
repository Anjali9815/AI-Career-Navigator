# retriever.py
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS FILE DOES:
#   1. Loads the ChromaDB that ingest.py built
#   2. Takes a user query + optional filters
#   3. Embeds the query → searches ChromaDB → returns top chunks
#   4. Formats those chunks for Claude's context window
#
# This file runs EVERY TIME a user asks a question.
# ingest.py runs once. retriever.py runs hundreds of times.
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
from typing import Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
COLLECTION_NAME = "career_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K           = 4
SCORE_THRESHOLD = 0.7     # cosine distance: 0=identical, 1=unrelated

# ── STEP 1: CONNECT TO CHROMADB ───────────────────────────────────────────────
# We don't re-embed anything here.
# We just LOAD the existing vectorstore from disk.
#
# ingest.py  → writes vectors to chroma_db/
# retriever.py → reads vectors from chroma_db/
#
# collection_metadata={"hnsw:space": "cosine"}
#   Forces ChromaDB to use cosine distance (0.0 - 1.0)
#   Without this it defaults to euclidean (l2) which gives scores > 1.0
#   MUST match what ingest.py used when creating the collection

def load_vectorstore() -> Chroma:
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DIR}. "
            f"Run ingest.py first to build it."
        )

    print(f"Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"Connecting to ChromaDB at {CHROMA_DIR}...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"}   # ← cosine not euclidean
    )

    count = vectorstore._collection.count()
    print(f"Connected. {count} vectors loaded.\n")
    return vectorstore


# ── STEP 2: CORE RETRIEVAL ────────────────────────────────────────────────────
# Called on EVERY user query.
#
# Parameters:
#   query       = user's question as plain text
#   vectorstore = loaded ChromaDB (load once, pass everywhere)
#   doc_type    = optional filter: "resume", "job_description", "general"
#                 None = search ALL documents
#   k           = how many chunks to return
#   threshold   = max cosine distance to accept
#                 lower = stricter (0.5 = only strong matches)
#                 higher = looser  (0.9 = accepts weak matches too)
#
# What happens internally:
#   1. query text → embedding model → 384-dim vector
#   2. ChromaDB computes cosine distance vs all stored vectors
#   3. Sort by distance, return top-k below threshold
#   4. Format into list of dicts with text + metadata + score

def retrieve(
    query: str,
    vectorstore: Chroma,
    doc_type: Optional[str] = None,
    k: int = TOP_K,
    threshold: float = SCORE_THRESHOLD
) -> list[dict]:
    """
    Core retrieval. Returns list of dicts:
    {
        "text"     : "chunk content",
        "source"   : "profile (1).pdf",
        "page"     : 0,
        "doc_type" : "resume",
        "score"    : 0.21,    ← cosine distance (lower = better match)
        "rank"     : 1        ← 1 = best match
    }

    SCORE GUIDE (after cosine fix):
        0.0 - 0.3  = very strong match
        0.3 - 0.5  = good match
        0.5 - 0.7  = moderate match
        0.7 - 1.0  = weak / reject
    """
    t0 = time.perf_counter()

    # ── Build metadata filter ─────────────────────────────────────────────
    # ChromaDB WHERE clause filters BEFORE similarity search
    # Only chunks matching the filter are searched
    #
    # Syntax: {"field": {"$eq": "value"}}
    # Other operators: $ne, $in, $nin, $gt, $gte, $lt, $lte
    #
    # Example:
    #   {"doc_type": {"$eq": "resume"}}
    #     → only search resume chunks
    #   {"doc_type": {"$in": ["resume", "job_description"]}}
    #     → search resumes AND job descriptions

    where_filter = {"doc_type": {"$eq": doc_type}} if doc_type else None

    if where_filter:
        print(f"Filter: doc_type = '{doc_type}'")

    # ── Similarity search with scores ─────────────────────────────────────
    # Returns List[Tuple[Document, float]]
    # float = cosine distance (with hnsw:space=cosine + normalize=True)
    # Range: 0.0 (identical) → 1.0 (completely unrelated)

    if where_filter:
        raw_results = vectorstore.similarity_search_with_score(
            query, k=k, filter=where_filter
        )
    else:
        raw_results = vectorstore.similarity_search_with_score(query, k=k)

    search_time = (time.perf_counter() - t0) * 1000

    # ── Threshold filter ──────────────────────────────────────────────────
    # Reject chunks that are too far from the query
    # Prevents Claude getting garbage context and hallucinating
    #
    # Without threshold: always returns k results no matter how irrelevant
    # With threshold:    returns 0 results if nothing is relevant
    #                    Claude then says "I couldn't find relevant info"

    filtered = [
        (doc, score) for doc, score in raw_results
        if score <= threshold
    ]

    # ── Format into clean dicts ───────────────────────────────────────────
    results = []
    for rank, (doc, score) in enumerate(filtered, start=1):
        results.append({
            "text"     : doc.page_content,
            "source"   : doc.metadata.get("filename", "unknown"),
            "page"     : doc.metadata.get("page", 0),
            "doc_type" : doc.metadata.get("doc_type", "general"),
            "score"    : round(score, 4),
            "rank"     : rank
        })

    print(f"Query    : '{query}'")
    print(f"Searched : {vectorstore._collection.count()} vectors")
    print(f"Raw      : {len(raw_results)} → after threshold: {len(filtered)}")
    print(f"Time     : {search_time:.2f}ms")
    if raw_results:
        scores = [round(s, 4) for _, s in raw_results]
        print(f"Scores   : {scores}")

    return results


# ── STEP 3: FORMAT FOR CLAUDE ─────────────────────────────────────────────────
# Converts retrieved chunks into a plain text string for Claude's prompt.
#
# Claude's prompt will look like:
#   "You are a career assistant. Use these documents to answer:
#    [CONTEXT ← this function's output]
#    Question: {user_query}"
#
# Why formatting matters:
#   Good format → Claude knows which source said what → accurate citations
#   Bad format  → Claude mixes up sources → hallucinations

def format_context(results: list[dict]) -> str:
    if not results:
        return "No relevant documents found."

    parts = []
    for r in results:
        header = (
            f"[Source {r['rank']}] {r['source']} "
            f"(page {r['page']}) | "
            f"type: {r['doc_type']} | "
            f"relevance score: {r['score']}"
        )
        parts.append(f"{header}\n{r['text']}")

    return "\n\n---\n\n".join(parts)


# ── STEP 4: CONVENIENCE WRAPPERS ─────────────────────────────────────────────
# Clean API for chain.py to call.
# Each returns (results, context):
#   results = list of dicts        ← used for citations in UI
#   context = formatted string     ← injected into Claude's prompt

def retrieve_all(query: str, vectorstore: Chroma) -> tuple[list, str]:
    """Search all documents regardless of type."""
    results = retrieve(query, vectorstore)
    return results, format_context(results)


def retrieve_from_resumes(query: str, vectorstore: Chroma) -> tuple[list, str]:
    """Search only resume documents."""
    results = retrieve(query, vectorstore, doc_type="resume")
    return results, format_context(results)


def retrieve_from_jobs(query: str, vectorstore: Chroma) -> tuple[list, str]:
    """Search only job description documents."""
    results = retrieve(query, vectorstore, doc_type="job_description")
    return results, format_context(results)


def match_resume_to_job(
    resume_query: str,
    job_query: str,
    vectorstore: Chroma
) -> tuple[str, str]:
    """
    Retrieves context from resumes AND job descriptions separately.
    Used in chain.py for: 'How well does this candidate match this role?'

    Returns (resume_context, job_context) as two separate strings.
    Claude gets both and compares them.
    """
    _, resume_context = retrieve_from_resumes(resume_query, vectorstore)
    _, job_context    = retrieve_from_jobs(job_query, vectorstore)
    return resume_context, job_context


# ── MAIN — test everything ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  AI Career Navigator — Retriever Test")
    print("=" * 55)

    vs = load_vectorstore()

    # ── Test 1: Basic retrieval ──
    print("\n[TEST 1] Basic retrieval — no filter")
    print("-" * 40)
    results, context = retrieve_all("Python machine learning experience", vs)
    print(f"\nTop result preview:\n{results[0]['text'][:200] if results else 'None'}")
    print(f"\nScore range: {[r['score'] for r in results]}")

    # ── Test 2: Resume filter ──
    print("\n\n[TEST 2] Resume-only filter")
    print("-" * 40)
    results, context = retrieve_from_resumes("FastAPI backend experience", vs)
    print(f"Results: {len(results)}")
    for r in results:
        print(f"  rank={r['rank']} score={r['score']} | {r['source']} p{r['page']}")

    # ── Test 3: Should return nothing ──
    print("\n\n[TEST 3] Unrelated query — expect 0 results")
    print("-" * 40)
    results, context = retrieve_all("quantum nuclear reactor engineering", vs)
    print(f"Results: {len(results)} (should be 0 or 1 max)")
    for r in results:
        print(f"  score={r['score']} | {r['source']} | {r['text'][:80]}")

    # ── Test 4: Full context string ──
    print("\n\n[TEST 4] Formatted context for Claude")
    print("-" * 40)
    results, context = retrieve_from_resumes("machine learning Python skills", vs)
    print(context[:600])

    print("\n" + "=" * 55)
    print("  All tests done. Ready for chain.py")
    print("=" * 55)