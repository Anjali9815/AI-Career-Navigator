import os
import warnings
from backend.src.vectorstore import final_production_search
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
from backend.src.extract import get_llm
RELEVANCE_THRESHOLD = -8.10

HARD_FLOOR = -10.0

def generate_context_prompt(query: str) -> str:
    raw_results = final_production_search(query, top_k=3)

    raw = final_production_search("who is good in knowledge graph", top_k=3)
    for doc, score in raw:
        print(score, doc.metadata["name"])


    if not raw_results:
        return None

    

    valid_results = [(doc, score) for doc, score in raw_results if score >= RELEVANCE_THRESHOLD]
    low_confidence = False
    if not valid_results:
        best_score = raw_results[0][1]
        if best_score >= HARD_FLOOR:
            valid_results = [raw_results[0]] 
            low_confidence = True
        else:
            return None
    

    profile_texts = []
    for rank, (doc, _) in enumerate(valid_results, start=1):
        separator = f"\n--- ALUMNI / PROFESSIONAL PROFILE PATHWAY #{rank} ---"
        profile_content = doc.page_content
        profile_texts.append(f"{separator}\n{profile_content}")

    context_string = "\n".join(profile_texts)

        # 4. PROMPT ASSEMBLY: Moves instructions to the bottom for maximum weight
    prompt = f"""You are an inspiring Career Navigator assistant for students. Your goal is to guide students by showing them real professional paths, skills, and background milestones achieved by alumni and industry experts in our database.

    ### CONTEXT: SUCCESSFUL CAREER PATHS
    {context_string}

    ### STUDENT'S QUESTION:
    {query}

    ### CRUCIAL GENERATION RULES (STRICT COMPLIANCE REQUIRED):
    1. STRICT WORD LIMIT: Your entire response must be 200 words max. Be concise, punchy, and scannable.
    2. NO TECHNOLOGY HALLUCINATION: Only list a technology, framework, language, or tool for a person if it explicitly appears inside that specific person's profile text block above. Do not assume or attach query technologies to a person's name by default.
    3. NO PERSONA MERGING: Evaluate and present each person's profile as a completely separate path. Never mix, merge, or cross-pollinate titles, companies, certifications, or milestones from Profile #1 into Profile #2. (e.g., do not mention another candidate's work history under a different name).
    4. NO SPECULATION: Rely strictly on facts. Never guess, extrapolate, or use speculative words like "likely", "probably", "presumably", "might", or "possibly". If it is not explicitly written in their context block, do not include it.
    5. If the context does not contain enough detailed information to answer the question, state cleanly that no matching career profiles were found in our database to illustrate this path.

    ### YOUR GUIDANCE AND NAVIGATIONAL RESPONSE:
    """
    return prompt




def ask_career_navigator(query: str) -> str:
    # 1. Build the context prompt string structure
    prompt = generate_context_prompt(query)

    # 2. API Firewall Gate check: If nothing survived the threshold, catch it immediately
    if prompt is None:
        return "I'm sorry, but no matching career profiles were found in our database to illustrate this path."

    # 3. Model Execution Stage: Run inference cleanly over the context
    llm = get_llm()
    response = llm.invoke(prompt)

    return response.content

if __name__ == "__main__":
    # valid_query = "worked on vue.js, c# and .net"
    garbage_query = "how to bake sourdough bread"
    new_query = "who is good in knowledge graph"

    # valid_prompt = generate_context_prompt(valid_query)
    # print(valid_prompt)

    # garbage_prompt = generate_context_prompt(garbage_query)
    # print(f"Result for garbage query: '{garbage_prompt}'")

    # print(ask_career_navigator(valid_query))
    # print("\n garbage", garbage_query)
    print(ask_career_navigator(garbage_query))
    print(ask_career_navigator(new_query))

