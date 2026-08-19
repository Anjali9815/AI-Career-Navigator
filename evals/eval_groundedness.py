import json
import os
import warnings
from langsmith import Client, evaluate

# Import your extraction model and the complete hybrid retrieval pipeline
from backend.src.extract import get_llm
from backend.src.vectorstore import final_production_search

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from backend.core.config import RELEVANCE_THRESHOLD, DATASET_NAME


client = Client()

judge_llm = get_llm()

def generate_context_prompt(query: str) -> dict:
    """Stage 1 & 2: Fetches hybrid matches, filters via threshold, and returns context."""
    raw_results = final_production_search(query, top_k=3)

    valid_results = [
        (doc, score) for doc, score in raw_results if score >= RELEVANCE_THRESHOLD
    ]

    # Return None if the query fails to clear the firewall gate
    if not valid_results:
        return None

    profile_texts = []
    for rank, (doc, _) in enumerate(valid_results, start=1):
        separator = f"\n--- ALUMNI / PROFESSIONAL PROFILE PATHWAY #{rank} ---"
        profile_texts.append(f"{separator}\n{doc.page_content}")

    context_string = "\n".join(profile_texts)

    # Note: Instructions sit at the absolute bottom for maximum weight focus
    prompt = f"""You are an inspiring Career Navigator assistant for students. Your goal is to guide students by showing them real professional paths, skills, and background milestones achieved by alumni and industry experts in our database.

### CONTEXT: SUCCESSFUL CAREER PATHS
{context_string}

### STUDENT'S QUESTION:
{query}

### CRUCIAL GENERATION RULES (STRICT COMPLIANCE REQUIRED):
1. STRICT WORD LIMIT: Your entire response must be 200 words max. Be concise, punchy, and scannable.
2. NO TECHNOLOGY HALLUCINATION: Only list a technology, framework, language, or tool for a person if it explicitly appears inside that specific person's profile text block above. Do not assume or attach query technologies to a person's name by default.
3. NO PERSONA MERGING: Evaluate and present each person's profile as a completely separate path. Never mix, merge, or cross-pollinate titles, companies, certifications, or milestones from Profile #1 into Profile #2.
4. NO SPECULATION: Rely strictly on facts. Never guess, extrapolate, or use speculative words like "likely", "probably", "presumably", "might", or "possibly".
5. If the context does not contain enough detailed information to answer the question, state cleanly that no matching career profiles were found in our database to illustrate this path.

### YOUR GUIDANCE AND NAVIGATIONAL RESPONSE:
"""
    return {"prompt": prompt, "raw_context": context_string}


def run_production_pipeline(inputs: dict) -> dict:
    """Stage 3: Executes inference over the contextualized student prompt."""
    query = inputs["query"]
    pipeline_data = generate_context_prompt(query)

    # Handle threshold-blocked queries immediately without wasting model tokens
    if pipeline_data is None:
        return {
            "response": "I'm sorry, but no matching career profiles were found in our database to illustrate this path.",
            "used_context": "",
        }

    llm = get_llm()
    response = llm.invoke(pipeline_data["prompt"])

    return {
        "response": response.content,
        "used_context": pipeline_data["raw_context"],
    }


# =====================================================================
# ⚖️ LLM-AS-A-JUDGE: FAITHFULNESS & GROUNDEDNESS EVALUATOR
# =====================================================================
def groundedness_judge(root_run, example) -> dict:
    """Decomposes the output response and cross-references it with source context."""
    generated_response = root_run.outputs.get("response")
    retrieved_context = root_run.outputs.get("used_context")

    if not generated_response:
        return {"key": "groundedness_score", "score": 0}

    # If the threshold firewall safely blocked the query, it is 100% grounded
    if not retrieved_context and "no matching career profiles" in generated_response:
        return {"key": "groundedness_score", "score": 1}

    verification_prompt = f"""
You are a meticulous quality assurance inspector tracking AI factual hallucinations. Your job is to verify whether the "Generated Assistant Response" is 100% grounded in the "Authoritative Source Context" provided.

Instructions:
1. Break down the Generated Assistant Response into distinct, standalone factual claims (e.g., individual statements about names, skills, job titles, companies, or universities).
2. Cross-check every single claim against the Authoritative Source Context text blocks.
3. If ANY claim states a fact, tool, tech framework, or job title that is NOT explicitly written in the Source Context, mark that claim as UNGROUNDED.
4. Output a brief, step-by-step justification detailing your findings, followed by a final score of exactly 1 if ALL claims are supported, or 0 if even a single fabrication exists.

### AUTHORITATIVE SOURCE CONTEXT:
{retrieved_context}

### GENERATED ASSISTANT RESPONSE:
{generated_response}

### EXECUTE YOUR STEP-BY-STEP VERIFICATION LOG AND OUTPUT FORMAT:
Reasoning Process: [List your claim extraction and source cross-check lines here]
Final Score: [Output exactly 1 or 0]
"""
    try:
        judge_output = judge_llm.invoke(verification_prompt).content

        if "Final Score: 1" in judge_output:
            score = 1
        elif "Final Score: 0" in judge_output:
            score = 0
        else:
            score = 1 if "1" in judge_output.split("Final Score:")[-1] else 0

        return {"key": "groundedness_score", "score": score}

    except Exception as e:
        print("Judge failed:", e)
        return {"key": "groundedness_score", "score": 0}


if __name__ == "__main__":
    print("\nTriggering automated LangSmith Groundedness evaluation suite...")

    experiment_result = evaluate(
        run_production_pipeline,
        data=DATASET_NAME,
        evaluators=[groundedness_judge],
        experiment_prefix="Production-Groundedness-v1",
    )
    print("\nEvaluation complete! View your groundedness scores on the dashboard.")
