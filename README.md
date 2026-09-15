# AI Career Navigator

Most career advice is generic. This matches a student to real alumni who already walked the path they want, and shows how those people got there.

Built for university students who don't know what to do next. Not AI generated suggestions, real paths that real people took.

---

## Table of contents

- [What it does](#what-it-does)
- [Tech stack](#tech-stack)
- [Why a standard RAG setup failed](#why-a-standard-rag-setup-failed)
- [Architecture](#architecture)
- [Pipeline stages](#pipeline-stages)
- [Evaluation](#evaluation)
- [Key design decisions](#key-design-decisions)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## What it does

A student describes their background and where they want to go. The system searches a corpus of alumni profiles, finds the people whose careers most closely match that trajectory, and returns a grounded summary of what those people actually did: which degrees, which companies, which skills, in what order.

Every claim in the answer traces back to a real profile. The system refuses rather than inventing a path when nothing in the corpus fits.

Current corpus: 21 LinkedIn profile exports.

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Orchestration | **LangChain** | retriever abstractions, ensemble fusion, structured output binding |
| PDF loading | **PyPDFLoader** via `DirectoryLoader` | `mode="single"` keeps each profile as one Document instead of splitting at page breaks |
| Extraction LLM | **Groq** (`openai/gpt-oss-120b`, `temperature=0`) | fast inference and a generous free tier; swapped in from Gemini after hitting a daily quota |
| Schema enforcement | **Pydantic** via `with_structured_output()` | the schema is enforced at the API level, so malformed JSON cannot be returned |
| Embeddings | **HuggingFaceEmbeddings**, `sentence-transformers/all-MiniLM-L6-v2` | see below |
| Vector store | **ChromaDB** | persists to disk, HNSW index, L2 distance |
| Keyword retrieval | **BM25Retriever** | catches exact tokens (company names, framework names) that embeddings blur |
| Fusion | **EnsembleRetriever** (reciprocal rank fusion) | 50/50 weighting of dense and sparse retrieval |
| Reranking | **CrossEncoder**, `cross-encoder/ms-marco-MiniLM-L-6-v2` | scores query and document jointly, producing comparable confidence scores |
| Generation LLM | **Groq** | grounded prompt with explicit anti hallucination rules |
| API | **FastAPI** + Uvicorn | auto generated docs at `/docs`, lifespan for startup loading |
| Evaluation | **LangSmith** | dataset versioning, experiment tracking, latency percentiles |

### On the embedding model

`all-MiniLM-L6-v2` is a **6 layer** transformer that produces **384 dimensional** embeddings. It is small enough to run comfortably on CPU with no GPU requirement, which makes it a practical default for semantic search and RAG pipelines where embedding latency sits on the critical path.

The tradeoff is that it is a **bi-encoder**: the query and the document are encoded independently and only compared afterwards, as two points in 384 dimensional space. That is what makes it fast enough to precompute an entire corpus, and also what makes it lose rare high signal tokens. The cross encoder in stage 2 exists to recover exactly that loss, by reading both texts together in a single pass.

---

## Why a standard RAG setup failed

The first version was a textbook RAG pipeline: load PDFs, split into 1000 character chunks, embed, retrieve, generate. It did not work, for three separate reasons.

**1. A chunk is not a person.**

Splitting profiles into 1000 character pieces produced 111 fragments across 21 people. A student matches against a whole career, not a paragraph from the middle of one. Retrieval would return "chunk 4 of Profile 8", which is a fragment of someone's job description, not a path anyone could follow.

Fix: extract each PDF into structured fields, build one text summary per person, and embed that. 21 vectors instead of 111 chunks.

**2. Pure vector search missed exact terms.**

Even with one vector per person, semantic search failed on the queries that mattered most. A query for knowledge graph experience returned the wrong person, while the profile that literally lists Neo4j, RDF Knowledge Graph and GraphRAG ranked lower. Queries naming a specific company ("who worked at Cognizant") missed the person who worked there.

Embeddings compress rare, high signal tokens into a general neighbourhood. Company names and framework names are exactly what gets blurred.

Fix: BM25 keyword retrieval running alongside vector retrieval, fused 50/50.

**3. Raw distances were not usable as confidence.**

Measured across a 20 case benchmark, nonsense queries frequently scored **better** than correct matches:

| Query | Type | L2 distance |
|---|---|---|
| Working as bus driver | nonsense | 1.35 |
| Professional football coaching | nonsense | 1.41 |
| worked on vue.js, c# and .net | correct match | 1.66 |
| robotics and computer vision | correct match | 1.71 |

No cutoff separates those two groups. Any threshold strict enough to block a bus driver also blocks legitimate matches.

Fix: a cross encoder reranker. Because it scores query and document as a pair rather than comparing two independently computed vectors, its scores are comparable across queries. That finally made a usable threshold possible.

---

## Architecture

```
                         ┌──────────────────────────┐
                         │  data/docs/*.pdf         │
                         │  LinkedIn profile export │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
  INGESTION         │  loader.py                         │
  (offline,         │  PyPDFLoader, mode="single"        │
   run once         │  one Document per PDF, not per page│
   per profile)     └─────────────────┬──────────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │  extract.py                        │
                    │  Groq LLM + Pydantic schema        │
                    │  with_structured_output()          │
                    │  incremental + resumable save      │
                    └─────────────────┬──────────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │  data/profiles.json      │
                         │  source of truth         │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
  ┌───────────▼────────────┐                     ┌────────────▼───────────┐
  │  profile_text.py       │                     │  BM25Retriever         │
  │  one summary string    │                     │  keyword index         │
  │  per person            │                     │  (in memory, startup)  │
  └───────────┬────────────┘                     └────────────┬───────────┘
              │                                               │
  ┌───────────▼────────────┐                                  │
  │  embedder.py           │                                  │
  │  all-MiniLM-L6-v2      │                                  │
  │  384 dimensions        │                                  │
  └───────────┬────────────┘                                  │
              │                                               │
  ┌───────────▼────────────┐                                  │
  │  Chroma                │                                  │
  │  HNSW index, L2 space  │                                  │
  │  persisted to disk     │                                  │
  └───────────┬────────────┘                                  │
              │                                               │
              └───────────────────┬───────────────────────────┘
                                  │
═══════════════════════════════════════════════════════════════════════
  QUERY TIME                      │
                                  │
                   ┌──────────────▼───────────────┐
                   │  student query               │
                   └──────────────┬───────────────┘
                                  │
                   ┌──────────────▼───────────────┐
                   │  STAGE 1: hybrid retrieval   │
                   │  EnsembleRetriever           │
                   │  vector 50% / BM25 50%       │
                   │  → 10 candidates             │
                   │  optimizes for RECALL        │
                   └──────────────┬───────────────┘
                                  │
                   ┌──────────────▼───────────────┐
                   │  STAGE 2: reranking          │
                   │  CrossEncoder                │
                   │  ms-marco-MiniLM-L-6-v2      │
                   │  scores (query, profile)     │
                   │  → top 3                     │
                   │  optimizes for PRECISION     │
                   └──────────────┬───────────────┘
                                  │
                   ┌──────────────▼───────────────┐
                   │  STAGE 3: confidence gate    │
                   │  score >= -8.1   → answer    │
                   │  -10.0 to -8.1   → weak match│
                   │  below -10.0     → refuse    │
                   └──────────────┬───────────────┘
                                  │
                   ┌──────────────▼───────────────┐
                   │  STAGE 4: generation         │
                   │  Groq + grounded prompt      │
                   │  anti hallucination rules    │
                   └──────────────┬───────────────┘
                                  │
                   ┌──────────────▼───────────────┐
                   │  FastAPI  POST /chat         │
                   └──────────────────────────────┘
```

---

## Pipeline stages

### 1. Loading

`PyPDFLoader` defaults to one Document per page, which cuts experience sections at page breaks. Loading with `mode="single"` collapses each PDF into one Document.

Measured effect: 57 page level Documents became 21 profile level Documents. Chunk boundary gaps dropped from 57 (one per page end) to 20 (one per file end), and every remaining gap sits correctly between two different people.

#### Loader comparison

LangChain offers several PDF loaders with different tradeoffs:

| Loader | Strengths | Limitations |
|---|---|---|
| **PyPDFLoader** | simple, reliable on text based PDFs; supports `mode` (`"single"` or `"page"`), `extract_images`, and password protected files | no OCR, so scanned or handwritten PDFs return nothing; flattens multi column layouts |
| **PDFMinerLoader** | better layout preservation, handles multi column documents such as resumes more faithfully | slower |
| **PyMuPDFLoader** | fastest; strong image and metadata handling, good for complex documents | AGPL licensed, which matters for commercial use |

This project uses `PyPDFLoader` with `mode="single"`. LinkedIn exports are text based, so OCR is not required.

The known cost of that choice is layout flattening. LinkedIn profiles are two column: a left sidebar carrying Contact, Top Skills, Languages, Certifications and Publications, and a main column carrying the headline, Summary, Experience and Education. `PyPDFLoader` emits the sidebar as a block at the top of the text, ahead of the person's name, with no marker separating it from the main content.

That directly shaped the extraction prompt, which has to describe the document layout explicitly so the model knows the three item "Top Skills" block is a sidebar field and not the person's full skill set. `PDFMinerLoader` would likely preserve that structure better and is the first thing to try if extraction quality becomes the bottleneck.

### 2. Extraction

Each profile PDF goes to an LLM with a Pydantic schema enforced through `with_structured_output()`. The schema is a hard constraint at the API level, not a prompt request, so malformed JSON cannot come back.

Extracted fields:

| Field | Notes |
|---|---|
| `name` | |
| `headline` | the one line self description, dense signal for matching |
| `summary` | the About section, the person's own account of their work |
| `skills` | collected from the Top Skills sidebar, the summary, and experience bullets |
| `languages` | sidebar |
| `certifications` | sidebar |
| `publications` | sidebar |
| `education[]` | degree, field, school, start, end |
| `experience[]` | title, company, start, end, description |

Two prompt rules came directly from observed failures:

- **Layout inheritance.** LinkedIn groups several roles under one company heading. Without an explicit rule, roles after the first came back with `company: null`.
- **Skills sourcing.** The Top Skills sidebar only ever lists three entries. Real technical skills live in the summary and the experience bullets, so the prompt names all three sources explicitly.

Extraction is **incremental and resumable**. Each record is written to disk immediately after it is produced, and already processed sources are skipped on rerun. This was not a design choice, it was learned: an early run saved only after the loop finished, hit a daily quota limit on profile 21, and lost all 20 successful extractions.

### 3. Profile to text

Structured records are flattened into one text block per person, covering headline, skills, certifications, summary, work history and education. This is the only thing that gets embedded, so whatever is omitted here can never be retrieved.

### 4. Embedding and storage

`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, stored in Chroma with an HNSW index using **L2 (Euclidean)** distance. Worth stating explicitly, because L2 distances are unbounded above 1.0, which makes them look wrong if you assume cosine.

### 5. Hybrid retrieval

`EnsembleRetriever` fuses Chroma vector search and BM25 keyword search 50/50 using reciprocal rank fusion, returning 10 candidates. This stage optimizes for recall: a profile the reranker never sees can never be returned.

### 6. Reranking

A cross encoder scores each (query, profile) pair directly. Unlike the bi-encoder used for embedding, it reads both texts in a single forward pass, so attention runs across the query and the document together. Slower per comparison, which is why it only runs on the 10 candidates rather than the full corpus.

Effect on the benchmark: queries that previously returned the wrong person at rank 1 (knowledge graph, CI/CD, Cognizant, robotics) all resolved to the correct person.

### 7. Confidence gate

Cross encoder scores are comparable across queries, so a threshold works here where it failed on raw distance.

| Band | Behaviour |
|---|---|
| `>= -8.1` | strong match, answer normally |
| `-10.0` to `-8.1` | weak match, answer with a low confidence caveat |
| `< -10.0` | refuse, state no matching profile was found |

### 8. Generation

The retrieved profiles plus the question go to the LLM with instructions placed **after** the context, since trailing instructions carry more weight over long inputs.

The generation rules each trace to an observed failure:

1. 200 word limit — the first version produced a 2000 word report nobody would read
2. No technology attribution unless it appears in that person's own profile block
3. No persona merging — one run attributed a different candidate's employer inside another person's work history
4. No speculation — "likely", "probably", "presumably" were being used to smuggle in guesses
5. Refuse cleanly when the context does not support an answer

---

## Evaluation

Three independent evals, tracked in LangSmith.

### Extraction completeness

Counts missing names, missing companies, missing start dates, and empty skills lists across all profiles, plus separate coverage counters for optional fields.

**Current: 0 errors across 21 profiles.**

Errors and coverage are reported as two separate numbers, because a missing publications list is data, not a defect. 18 of 21 people have no publications.

*What it does not measure:* correctness. A company field populated with the **wrong** company scores identically to a right one. This metric also rewards hallucination, since a model that invents values scores perfectly. That is the gap the groundedness eval exists to cover.

### Retrieval accuracy

A labelled benchmark of 20 cases in `data/retrieval_tests.json`. Each case carries a query, a `should_match` flag, and the expected names. Scoring checks both whether an expected name was returned and whether the confidence gate behaved correctly.

**Current: 0.79 correct_retrieval.**
**Latency: P50 1.4s, P99 1.8s** (dominated by the cross encoder).

*What it does not measure:* generalization. 20 cases is a small benchmark, and two of them ("who knows Chinese", "domain experience in finance") are unanswerable from the corpus because that information is not present in the source PDFs. They are retained deliberately as a reminder that retrieval cannot recover what extraction never captured.

### Groundedness

LLM as judge. The generated answer and the exact retrieved context are sent to a second model, which decomposes the answer into individual factual claims and checks each against the source, returning a binary grounded or not grounded verdict.

*What it does not measure:* anything the judge itself gets wrong. The judge is a language model with the same failure modes as the thing it is judging. It catches obvious fabrication reliably and subtle misattribution less so.

---

## Key design decisions

**Structured extraction rather than chunking.** The unit of retrieval has to be the unit the user is looking for. A student is looking for a person, so the index stores one vector per person.

**Prompting cannot enforce grounding.** Three increasingly explicit prompt revisions did not stop the model from attributing one candidate's work to another. Only verification catches it. Prompts request, code verifies.

**Thresholds must be measured, not chosen.** The first threshold was picked before any measurement and would have rejected every correct answer in the benchmark. The working threshold came from plotting the true and false score distributions and finding the gap between them.

**Provider swap is a one function change.** `get_llm()` is the only place a provider is named. The project moved from Gemini to Groq after hitting a free tier daily quota, and nothing else in the codebase changed.

**Both indexes must be rebuilt together.** Chroma reads from disk, BM25 rebuilds from `profiles.json` at startup. Rebuilding one and not the other produces a system that returns confidently wrong results with no error. This caused a real false negative during development: a query for knowledge graph experience returned nothing while the matching profile sat in the JSON, because the vector index was stale.

---

## Project structure

```
AI-Career-Navigator/
├── backend/
│   ├── main.py                 FastAPI app, lifespan, endpoints
│   ├── core/
│   │   ├── config.py           all paths computed in one place
│   │   └── logger.py
│   └── src/
│       ├── loader.py           PDF loading and splitting
│       ├── extract.py          LLM extraction, Pydantic schema
│       ├── profile_text.py     record to embeddable text
│       ├── embedder.py         embedding model
│       ├── vectorstore.py      Chroma build, hybrid search, rerank
│       └── chain.py            prompt assembly and generation
├── evals/
│   ├── score_extraction.py     extraction completeness
│   ├── run_eval.py             retrieval accuracy (LangSmith)
│   ├── threshold_test.py       raw score distribution
│   └── eval_groundedness.py    LLM as judge
├── data/
│   ├── docs/                   source PDFs
│   ├── profiles.json           extracted records, source of truth
│   └── retrieval_tests.json    labelled benchmark
├── frontend/
├── chroma_db/                  generated, gitignored
├── .env                        gitignored
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` at the project root:

```
GROQ_API_KEY=your_key
LANGSMITH_API_KEY=your_key
```

Build the index from source PDFs:

```bash
python -m backend.src.extract       # PDFs  → profiles.json
python -m backend.src.vectorstore   # JSON  → chroma_db/
```

Run the API:

```bash
uvicorn backend.main:app --reload
```

Open `http://localhost:8000/docs` and post a query to `/chat`.

Run the evals:

```bash
python -m evals.score_extraction
python -m evals.run_eval
python -m evals.eval_groundedness
```

---

## Known limitations

**Corpus size.** 21 profiles, all drawn from one university network. The retrieval techniques are scale independent, but the benchmark numbers are not: 20 test cases over 21 documents is a small sample, and the threshold gap is narrow enough that it will need retuning as the corpus grows.

**BM25 is not incremental.** `BM25Retriever` builds its index in memory at startup with no append operation. Adding one profile rebuilds the whole index. At this scale that is milliseconds. At scale the keyword half would move to Elasticsearch or OpenSearch.

**Chroma writes are full rebuilds.** The current build uses `from_texts()`, which recreates the collection. `add_texts()` with explicit ids would make additions incremental, and is the next optimization.

**Two indexes, two sources of truth.** Chroma and BM25 must be rebuilt together or they drift. There is no guard against this today beyond discipline.

**Extraction is capped by model token budget.** One profile exceeded the tokens per minute limit on the free tier and had to be rerun with a lower output cap. Long documents would need section by section extraction or a larger budget.

**No auth, no upload, no persistence of user data.** Matching runs against a fixed corpus.

**Certifications and experience are not distinguished at generation time.** A person who completed a course in a technology and a person who shipped production systems in it both surface as having that skill.

---

## Roadmap

- `POST /match` — accept a resume PDF plus a target role, extract it on the fly, and use the combined text as the query
- Incremental writes with `add_texts()` and stable ids
- A `type` field (`alumni` / `student`) in metadata, with filtered retrieval so students are matched against completed paths rather than other beginners
- Promotion rule: a stored student whose latest education end date has passed and who holds a full time role becomes an alumni path
- Frontend: upload, goal input, matched profiles, roadmap view
- Separate "learned" from "did" when presenting skills