# AI-Career-Navigator


ai-career-navigator/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── ingest.py            # PDF loading + chunking + ChromaDB
│   ├── retriever.py         # Query embedding + similarity search
│   ├── chain.py             # LangChain + Claude prompt chain
│   └── requirements.txt
├── frontend/
│   └── app.py               # Streamlit chat UI
├── data/
│   └── docs/                # Drop your PDFs here
└── chroma_db/               # Auto-created vector store



LangChain provides three main PDF loaders, each suited for different needs.
PyPDFLoader works best with text-based PDFs. If the PDF contains scanned images or handwritten text, it won't extract content since it doesn't include OCR capabilities.
PyPDFLoader supports a mode parameter — either "single" for the entire document as one chunk, or "page" for page-wise extraction. It also supports extract_images and password-protected PDFs.

PDFMinerLoader — better layout preservation, good for resumes with columns

PyMuPDFLoader — fastest, also handles images and metadata well. Good if your PDFs are complex.


PyPDFLoader - text based pdf
PDFMinerLoader - column based pdf
PuMUPDFLoader - image/text based pdf



"Python developer with ML experience"  → [0.23, 0.87, -0.12, 0.45, ...]  (384 numbers)
"Software engineer skilled in AI"      → [0.21, 0.85, -0.10, 0.43, ...]  (very close!)
"I love eating pizza"                  → [0.91, -0.34, 0.67, -0.22, ...]  (far away)


Your question → embed → query vector
                              ↓
              ChromaDB: find top-4 closest vectors
                              ↓
              Return their text + metadata
                              ↓
              Stuff into Claude's prompt as context






HuggingFaceEmbeddings -all-MiniLM-L6-v2 model - 384 dimension
it generates 384-dimensional embeddings. It’s lightweight and efficient, making it ideal for semantic search and RAG pipelines, especially when running on CPU.”
It use 6 transformer layer


similarity = dot(A, B) / (|A| × |B|)

Where:
  A = query vector    [0.21, -0.85, 0.43, ...]
  B = chunk vector    [0.23, -0.87, 0.45, ...]

dot(A,B) = (0.21×0.23) + (-0.85×-0.87) + (0.43×0.45) + ...
         = 0.048 + 0.739 + 0.193 + ...
         = high number ← vectors point same direction = SIMILAR

Result: similarity = 0.94  ← very close to 1.0 = strong match


