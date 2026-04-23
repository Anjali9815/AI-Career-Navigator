# chain.py
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS FILE DOES:
#   1. Takes user query + retrieved context from retriever.py
#   2. Builds a prompt (system + context + question)
#   3. Sends to Google Gemini 1.5 Flash (FREE)
#   4. Returns answer + source citations
#
# LLM: Google Gemini 1.5 Flash
#   - Free tier: 15 requests/min, 1500 requests/day
#   - Fast response time
#   - No credit card needed
#   - Get key: aistudio.google.com
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from retriever import (
    load_vectorstore,
    retrieve_all,
    retrieve_from_resumes,
    retrieve_from_jobs,
    match_resume_to_job,
    format_context
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# GEMINI_MODEL = "gemini-2.0"   # free tier model
GEMINI_MODEL = "gemini-2.5-flash"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.2   # low = more factual, less creative
               # 0.0 = deterministic (same answer every time)
               # 1.0 = creative/varied

# ── GEMINI CLIENT ─────────────────────────────────────────────────────────────
# Reads GOOGLE_API_KEY from environment
# Set once: export GOOGLE_API_KEY="your-key-here"
#
# ChatGoogleGenerativeAI wraps Gemini in LangChain's standard interface
# This means if you later switch to Claude or GPT, only this line changes

def get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # ← hardcoded, not variable
        google_api_key=api_key,
        max_output_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )


# ── PROMPT TEMPLATES ──────────────────────────────────────────────────────────
# TWO parts to every LLM prompt:
#
# 1. SYSTEM MESSAGE — tells the LLM who it is and the rules
#    Applied once, affects all responses
#
# 2. HUMAN MESSAGE — the actual question + retrieved context
#    Built fresh every query
#
# WHY THIS MATTERS FOR RAG:
#   Without "answer only from documents" → LLM uses training data → hallucination
#   With the rule → LLM stays grounded in YOUR documents → accurate answers

SYSTEM_PROMPT = """You are an expert AI Career Navigator assistant.

You help users analyze candidate profiles, identify skills, compare 
candidates, and provide career insights based on real LinkedIn profile documents.

STRICT RULES:
- Answer ONLY using information from the provided documents
- Always cite which profile/source the information came from
- If documents don't contain the answer, say: "The provided profiles don't mention this"
- Be specific — mention actual skills, companies, and experience from the docs
- Structure your answers clearly with bullet points where helpful
- For comparisons, be explicit about who has what"""


def build_prompt(query: str, context: str) -> str:
    """
    Builds the human message with context embedded.

    Why inject context into the human message (not system)?
        Context changes every query — it's dynamic
        System prompt is static — same for every query
        Keeps separation clean
    """
    return f"""Here are the relevant profile documents:
{context}

Based on these documents, please answer:
{query}"""


# ── SINGLE-TURN CHAIN ─────────────────────────────────────────────────────────
# One question → retrieve → ask Gemini → one answer
# No memory between calls
#
# Use this for:
#   - Simple lookups: "who has Python experience?"
#   - One-shot analysis: "summarize all profiles"
#   - API endpoints that don't need conversation history

def run_chain(
    query: str,
    vectorstore,
    mode: str = "all"
) -> dict:
    """
    Full RAG chain: retrieve → prompt → Gemini → response

    mode options:
        "all"    → search all documents
        "resume" → search only resumes
        "job"    → search only job descriptions
        "match"  → compare resume vs job description

    Returns:
        {
            "answer"  : Gemini's response,
            "sources" : list of files used,
            "chunks"  : raw retrieved chunks,
            "context" : formatted context sent to Gemini,
            "tokens"  : usage metadata
        }
    """

    # ── Step 1: Retrieve ──────────────────────────────────────────────────
    print(f"\nMode: {mode} | Query: '{query[:60]}...'")

    if mode == "resume":
        results, context = retrieve_from_resumes(query, vectorstore)
    elif mode == "job":
        results, context = retrieve_from_jobs(query, vectorstore)
    elif mode == "match":
        resume_ctx, job_ctx = match_resume_to_job(query, query, vectorstore)
        context = f"RESUME PROFILES:\n{resume_ctx}\n\nJOB REQUIREMENTS:\n{job_ctx}"
        results = []
    else:
        results, context = retrieve_all(query, vectorstore)

    # ── Step 2: No results guard ──────────────────────────────────────────
    # Never call LLM with empty context
    # It would hallucinate an answer from thin air

    if context == "No relevant documents found.":
        return {
            "answer"  : "I couldn't find relevant information in the loaded profiles for your query.",
            "sources" : [],
            "chunks"  : [],
            "context" : context,
            "tokens"  : {}
        }

    # ── Step 3: Build messages ────────────────────────────────────────────
    # LangChain message format:
    #   SystemMessage → sets behavior
    #   HumanMessage  → user's question + context
    #
    # Gemini receives these as a conversation turn

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(query, context))
    ]

    # ── Step 4: Call Gemini ───────────────────────────────────────────────
    print(f"Calling Gemini ({GEMINI_MODEL})...")
    print(f"Context: {len(context)} chars | Prompt: {len(build_prompt(query, context))} chars")

    llm = get_llm()
    response = llm.invoke(messages)

    # response.content = the answer text
    # response.response_metadata = token usage, model info
    answer = response.content
    metadata = response.response_metadata

    print(f"Done. Response length: {len(answer)} chars")

    # ── Step 5: Extract sources ───────────────────────────────────────────
    sources = list(set(r["source"] for r in results)) if results else []

    return {
        "answer"  : answer,
        "sources" : sources,
        "chunks"  : results,
        "context" : context,
        "tokens"  : metadata
    }


# ── MULTI-TURN CHAT CHAIN ─────────────────────────────────────────────────────
# Maintains conversation history across multiple questions.
#
# WHY THIS IS NEEDED:
#   Turn 1: "Who has AWS experience?"
#     → Gemini answers with 3 candidates
#   Turn 2: "Which one is most senior?"
#     → Without history: Gemini has no idea who "one" refers to
#     → With history: Gemini remembers turn 1 and answers correctly
#
# HOW IT WORKS:
#   message_history grows with each turn:
#   [
#     HumanMessage("Who has AWS..."),
#     AIMessage("Profile 3 and 7 have AWS..."),
#     HumanMessage("Which is most senior?"),   ← new turn
#   ]
#   Full history sent to Gemini every call
#
# NOTE: Fresh context retrieved each turn
#   "Which is more senior?" → retrieves relevant chunks for this question
#   This ensures each answer is grounded in documents

def run_chat_chain(
    query: str,
    vectorstore,
    message_history: list,
    mode: str = "all"
) -> tuple[str, list]:
    """
    Multi-turn RAG chain with conversation memory.

    message_history: list of LangChain message objects
        Starts empty: []
        Grows with each turn
        Pass the same list back each time

    Returns (answer, updated_message_history)

    Usage:
        history = []
        answer1, history = run_chat_chain("Q1", vs, history)
        answer2, history = run_chat_chain("Q2", vs, history)
        # Gemini remembers Q1 when answering Q2
    """

    # Retrieve fresh context for this query
    if mode == "resume":
        results, context = retrieve_from_resumes(query, vectorstore)
    elif mode == "job":
        results, context = retrieve_from_jobs(query, vectorstore)
    else:
        results, context = retrieve_all(query, vectorstore)

    # Add new question to history
    user_message = HumanMessage(content=build_prompt(query, context))
    message_history.append(user_message)

    # Build full message list: system + full history
    all_messages = [SystemMessage(content=SYSTEM_PROMPT)] + message_history

    # Call Gemini with full conversation
    llm = get_llm()
    response = llm.invoke(all_messages)
    answer = response.content

    # Add Gemini's response to history for next turn
    message_history.append(AIMessage(content=answer))

    return answer, message_history


# ── MAIN — test everything ────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  AI Career Navigator — Chain Test (Gemini)")
    print("=" * 60)

    # Load vectorstore once — reused for all queries
    vs = load_vectorstore()

    # ── Test 1: Find candidates by skill ──
    print("\n[TEST 1] Who has Python + ML experience?")
    print("-" * 50)
    result = run_chain(
        "Who has the strongest Python and machine learning experience? List their key skills.",
        vs,
        mode="resume"
    )
    print(f"\nGemini's answer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")

    # ── Test 2: Specific technology ──
    print("\n\n[TEST 2] Find NLP candidates")
    print("-" * 50)
    result = run_chain(
        "Which candidates have NLP or natural language processing experience?",
        vs,
        mode="resume"
    )
    print(f"\nGemini's answer:\n{result['answer']}")

    # ── Test 3: Multi-turn conversation ──
    print("\n\n[TEST 3] Multi-turn chat")
    print("-" * 50)
    history = []

    a1, history = run_chat_chain(
        "Who has cloud platform experience (AWS, GCP, or Azure)?",
        vs, history, mode="resume"
    )
    print(f"Turn 1:\n{a1}\n")

    a2, history = run_chat_chain(
        "Of those candidates, who has the most overall experience?",
        vs, history, mode="resume"
    )
    print(f"Turn 2 (follow-up, uses memory):\n{a2}")
    print(f"\nConversation history: {len(history)} messages")

    print("\n" + "=" * 60)
    print("  Chain working! Ready for main.py (FastAPI)")
    print("=" * 60)