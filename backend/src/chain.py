import json
import warnings
from typing import List, Optional

from pydantic import BaseModel, Field

from backend.src.extract import get_llm
from backend.src.timeline import profile_for, timeline_for
from backend.src.vectorstore import final_production_search

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

RELEVANCE_THRESHOLD = -8.10
HARD_FLOOR = -10.0


# ── RESPONSE SCHEMA ───────────────────────────────────────────────────────────
# Note what is NOT here: `path` and `linkedin_url`. Career steps are sorted in
# code from profiles.json (see timeline.py), and the LinkedIn URL is looked up
# by name. The model was returning career steps out of chronological order, and
# a URL is exactly the kind of value a model will happily invent. What the model
# cannot generate, it cannot get wrong.

class MatchedProfile(BaseModel):
    name: str = Field(
        description="Full name of the alumni, exactly as written in their profile block."
    )
    current_role: Optional[str] = Field(
        None,
        description="Their most recent title and company, e.g. 'Data Analyst at WikiCharities'. Null if not stated.",
    )
    skills: List[str] = Field(
        default=[],
        description=(
            "Up to 8 skills from this person's profile block that are relevant to "
            "the student's question. Skills are technologies, tools, or capabilities "
            "only. Never job titles, company names, or role names."
        ),
    )
    why_relevant: str = Field(
        description="One sentence on why this person's path matches what the student asked. Facts only."
    )


class NavigatorResponse(BaseModel):
    direction: str = Field(
        description=(
            "Two sentences naming the field or role direction that fits the student, "
            "grounded in what the matched profiles actually did."
        )
    )
    matches: List[MatchedProfile] = Field(
        default=[],
        description="One entry per profile in the context. Never invent a person who is not in the context.",
    )
    next_steps: List[str] = Field(
        default=[],
        description=(
            "3 to 5 concrete actions the student can take, each one drawn from a step "
            "the matched people actually took. One short line each."
        ),
    )


# ── STAGE 1 & 2: RETRIEVAL AND CONTEXT ASSEMBLY ───────────────────────────────

def generate_context_prompt(query: str):
    """
    Returns a dict with the assembled prompt and a low confidence flag,
    or None when nothing clears the hard floor.
    """
    raw_results = final_production_search(query, top_k=3)

    if not raw_results:
        return None

    valid_results = [
        (doc, score) for doc, score in raw_results if score >= RELEVANCE_THRESHOLD
    ]

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
        profile_texts.append(f"{separator}\n{doc.page_content}")

    context_string = "\n".join(profile_texts)

    confidence_line = ""
    if low_confidence:
        confidence_line = (
            "\nNOTE: This is the closest available profile, not a strong match. "
            "Say so plainly in the direction field.\n"
        )

    # Instructions sit at the bottom: trailing instructions carry more weight
    # over long inputs.
    prompt = f"""You are a Career Navigator assistant for university students. Your goal is to show students real professional paths taken by alumni in our database, so they can follow a proven route rather than generic advice.

### CONTEXT: ALUMNI CAREER PATHS
{context_string}
{confidence_line}
### STUDENT'S QUESTION:
{query}

### CRUCIAL GENERATION RULES (STRICT COMPLIANCE REQUIRED):
1. ONE ENTRY PER PROFILE: Produce exactly one match entry for each profile block in the context above. Never invent a person who does not appear there.
2. NAMES MUST BE EXACT: Copy each name character for character from its profile block. The name is used to look up their career timeline and LinkedIn URL, so an altered name silently breaks both.
3. NO TECHNOLOGY HALLUCINATION: Only list a technology, framework, language, or tool for a person if it explicitly appears inside that specific person's profile block. Do not attach the student's query technologies to a person by default.
4. NO PERSONA MERGING: Each entry describes one person only. Never move a title, company, certification, or milestone from one profile block into another person's entry.
5. NO SPECULATION: Facts only. Never use "likely", "probably", "presumably", "might", or "possibly". If it is not written in their block, leave the field out.
6. CERTIFICATIONS ARE NOT EXPERIENCE: A course or certification in a technology means they studied it. Do not present it as work experience.
7. KEEP IT SHORT: Every line is one short phrase, not a paragraph.

### YOUR STRUCTURED RESPONSE:
"""
    return {"prompt": prompt, "low_confidence": low_confidence}



def clean_url(url):
    if not url:
        return None
    url = url.strip()
    # The model sometimes returns markdown: [text](https://...)
    if "](" in url:
        url = url.split("](", 1)[1].rstrip(")")
    url = url.replace("https://", "").replace("http://", "").strip("[]() ")
    return url or None

# ── STAGE 3: GENERATION ───────────────────────────────────────────────────────

def _empty_response(message):
    return {
        "direction": "",
        "matches": [],
        "next_steps": [],
        "low_confidence": False,
        "no_match": True,
        "message": message,
    }


def enrich_matches(matches):
    """
    Attach the timeline and the LinkedIn URL by looking each person up in
    profiles.json by name. Both lookups degrade quietly: an unresolved name
    yields an empty timeline and no link rather than raising, so one bad name
    costs a card's detail instead of the whole response.
    """
    for match in matches:
        name = match.get("name")
        record = profile_for(name) or {}
        match["timeline"] = timeline_for(name)
        match["linkedin_url"] = clean_url(record.get("linkedin_url"))
    return matches


def ask_career_navigator(query: str) -> dict:
    pipeline_data = generate_context_prompt(query)

    if pipeline_data is None:
        return _empty_response(
            "No matching career profiles were found in our database to illustrate this path."
        )

    llm = get_llm()
    structured_llm = llm.with_structured_output(NavigatorResponse)

    try:
        result = structured_llm.invoke(pipeline_data["prompt"])
    except Exception as e:
        print("Generation failed:", e)
        return _empty_response(
            "Something went wrong while building your match. Please try again."
        )

    payload = result.model_dump()

    enrich_matches(payload.get("matches", []))

    payload["low_confidence"] = pipeline_data["low_confidence"]
    payload["no_match"] = False
    payload["message"] = ""
    return payload


if __name__ == "__main__":
    queries = [
        "how to bake sourdough bread",
        "who is good in knowledge graph",
        "I studied electronics engineering and want to move into data analysis",
    ]

    for q in queries:
        print("\n" + "=" * 70)
        print("QUERY:", q)
        print("=" * 70)
        print(json.dumps(ask_career_navigator(q), indent=2))