# ============================================================
# ingest.py — AI Career Navigator
# ============================================================
# WHAT THIS FILE DOES (in order):
#   1. Loads all PDFs/text files from data/docs/
#   2. Splits them into chunks (500 chars, 50 overlap)
#   3. Embeds each chunk into a vector
#   4. Stores vectors + text + metadata in ChromaDB
#
# Run this ONCE to build your vector store.
# Re-run whenever you add new documents.
# ============================================================

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DOCS_DIR = "data/docs"          # Drop your PDFs/txts here
CHROMA_DIR = "chroma_db"        # ChromaDB will be created here
COLLECTION_NAME = "career_docs" # Name of the collection inside ChromaDB
CHUNK_SIZE = 500                # Max characters per chunk
CHUNK_OVERLAP = 50              # Shared characters between adjacent chunks
TOP_K = 4                       # How many chunks to retrieve at query time

# ─────────────────────────────────────────
# STEP 1: LOAD DOCUMENTS
# ─────────────────────────────────────────
# DirectoryLoader scans the folder and routes each file
# to the right loader based on extension.
# Each page of a PDF becomes one Document object with metadata:
#   { source: "data/docs/resume.pdf", page: 0 }

def load_documents(docs_dir: str) -> list:
    print(f"\nLoading documents from: {docs_dir}")

    # Load PDFs — one Document per page
    pdf_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True   # loads multiple PDFs in parallel
    )

    pdf_docs = pdf_loader.load()
    
    print(f"PDFs: {len(pdf_docs)} pages")
    return pdf_docs


# ─────────────────────────────────────────
# STEP 2: ADD METADATA
# ─────────────────────────────────────────
# We tag each document with a doc_type so we can filter later.
# Example: "Give me only job descriptions" or "Search only resumes"
# 
# This is what makes your RAG non-basic — metadata filtering.
# Without this, every query searches ALL documents blindly.

def tag_metadata(docs: list) -> list:
    for doc in docs:
        source = doc.metadata.get("source", "").lower()
        filename = doc.metadata.get("filename", "").lower()

        # Debug — print every file being tagged
        print(f"  Tagging: {source}")

        if any(kw in source for kw in ["resume", "cv", "profile"]):
            doc.metadata["doc_type"] = "resume"
        elif any(kw in source for kw in ["jd", "job", "description"]):
            doc.metadata["doc_type"] = "job_description"
        else:
            doc.metadata["doc_type"] = "general"

        doc.metadata["filename"] = os.path.basename(source)

    # Print summary
    from collections import Counter
    counts = Counter(d.metadata["doc_type"] for d in docs)
    print(f"Tag summary: {dict(counts)}")
    return docs

# ─────────────────────────────────────────
# STEP 3: SPLIT INTO CHUNKS
# ─────────────────────────────────────────
# RecursiveCharacterTextSplitter tries splitting in this order:
#   1. Double newline (\n\n) — paragraph boundary  ← tries this first
#   2. Single newline (\n)   — line boundary
#   3. Period (.)            — sentence boundary
#   4. Space ( )             — word boundary
#   5. Empty string ("")     — character boundary  ← last resort
#
# This means it RESPECTS natural language structure.
# It only goes to the next separator if the current chunk is still > chunk_size.
#
# MATH EXAMPLE:
#   doc = 1200 chars
#   chunk_size = 500, overlap = 50
#
#   Chunk 1: chars   0 → 500   (500 chars)
#   Chunk 2: chars 450 → 950   (500 chars, starts 50 back = overlap)
#   Chunk 3: chars 900 → 1200  (300 chars, last chunk)
#   Total: 3 chunks

def split_documents(docs: list) -> list:
    print(f"\n✂️  Splitting documents...")
    print(f"   Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,      # counts characters (not tokens)
    )

    chunks = splitter.split_documents(docs)

    # MATH: show chunking stats
    total_chars = sum(len(c.page_content) for c in chunks)
    avg_chars = total_chars // len(chunks) if chunks else 0

    print(f"✅ Created {len(chunks)} chunks")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Avg chunk size:   {avg_chars} chars")
    print(f"   Sample chunk:\n   ---\n   {chunks[0].page_content[:200]}\n   ---")

    
    return chunks


# ─────────────────────────────────────────
# STEP 4: CREATE EMBEDDINGS MODEL
# ─────────────────────────────────────────
# We use sentence-transformers/all-MiniLM-L6-v2
# This is a FREE, local model — no API key needed.
#
# What it does:
#   "Anjali has 3 years Python experience"
#    → [0.23, -0.87, 0.45, 0.12, ... ] ← 384 numbers
#
# Each number = one dimension of meaning in 384D space.
# Semantically similar sentences → vectors close together.
# Cosine similarity measures how "close" two vectors are:
#   similarity = dot(A, B) / (|A| * |B|)
#   1.0 = identical meaning, 0.0 = unrelated

def get_embeddings_model():
    print(f"\n🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True}  # needed for cosine similarity
    )
    print(f"✅ Embedding model loaded (384 dimensions per chunk)")
    return embeddings


# ─────────────────────────────────────────
# STEP 5: STORE IN CHROMADB
# ─────────────────────────────────────────
# ChromaDB stores three things per chunk:
#   1. The vector  (384 floats)              ← for similarity search
#   2. The text    (original chunk content)  ← returned to LLM
#   3. The metadata (source, page, doc_type) ← for filtering
#
# It persists to disk at CHROMA_DIR so you don't re-embed every run.
#
# IMPORTANT: Chroma.from_documents() embeds ALL chunks in one call.
# For 197 chunks × 384 dims = 75,648 floats stored.


def store_in_chromadb(chunks: list, embeddings) -> Chroma:
    print(f"\n💾 Storing {len(chunks)} chunks in ChromaDB...")
    print(f"   Location: {CHROMA_DIR}/")

    # If collection already exists, delete and recreate
    # (so re-running ingest doesn't duplicate documents)
    import shutil
    if os.path.exists(CHROMA_DIR):
        print(f"   ⚠️  Existing ChromaDB found — clearing it first")
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"}  # ← ADD THIS
    )

    # print(f"✅ ChromaDB created at ./{CHROMA_DIR}/")
    # print(f"   Collection: {COLLECTION_NAME}")
    # print(f"   Vectors stored: {vectorstore._collection.count()}")
    return vectorstore


# ─────────────────────────────────────────
# STEP 6: TEST RETRIEVAL
# ─────────────────────────────────────────
# Run a quick sanity check — query the vectorstore
# and see what comes back. This confirms everything worked.

def test_retrieval(vectorstore: Chroma):
    print(f"\n🔍 Testing retrieval...")
    test_query = "machine learning experience"

    # Basic similarity search — returns top-k chunks
    results = vectorstore.similarity_search(test_query, k=TOP_K)

    print(f"   Query: '{test_query}'")
    print(f"   Retrieved {len(results)} chunks:\n")
    for i, doc in enumerate(results):
        print(f"   [{i+1}] Source: {doc.metadata.get('filename', 'unknown')}")
        print(f"        Type:   {doc.metadata.get('doc_type', 'unknown')}")
        print(f"        Text:   {doc.page_content[:120]}...")
        print()

    # Advanced: similarity search WITH scores
    # Score is cosine distance — lower = more similar (0.0 = identical)
    print(f"   Scores (cosine distance — lower = better match):")
    scored = vectorstore.similarity_search_with_score(test_query, k=TOP_K)
    for doc, score in scored:
        print(f"   score={score:.4f} | {doc.metadata.get('filename')} | {doc.page_content[:80]}...")


# ─────────────────────────────────────────
# MAIN — runs all steps in order
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  AI Career Navigator — Document Ingestion Pipeline")
    print("=" * 55)

    # Check docs folder exists
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)
        print(f"⚠️  Created empty {DOCS_DIR}/ — add your PDFs there and re-run")
        exit(0)

    # Run the full pipeline
    docs       = load_documents(DOCS_DIR)
    docs       = tag_metadata(docs)
    chunks     = split_documents(docs)
    embeddings = get_embeddings_model()
    vectorstore = store_in_chromadb(chunks, embeddings)
    test_retrieval(vectorstore)

    print("\n✅ Ingestion complete! Ready to build retriever.py next.")