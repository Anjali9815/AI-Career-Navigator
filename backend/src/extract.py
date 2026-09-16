import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from backend.core.config import DOCS_DIR, PROFILES_PATH
from backend.core.logger import get_logger
from backend.src.loader import load_pdf

OUTPUT_FILE = PROFILES_PATH
load_dotenv()

log = get_logger(__name__)

# Seconds to wait between profiles. Groq's free tier caps tokens per minute,
# and descriptions made each request large enough to hit it regularly.
THROTTLE_SECONDS = 3

# Retries on a rate limit before giving up on a profile.
MAX_RETRIES = 3


# ── SCHEMA ────────────────────────────────────────────────────────────────────
# Enforced at the API level through with_structured_output(), so malformed
# JSON cannot come back. Field descriptions do real work here: the model reads
# them, so each one is written as an instruction rather than a label.

class ExperienceItem(BaseModel):
    title: Optional[str] = Field(None, description="Job title. Use null if missing.")
    company: Optional[str] = Field(None, description="Company name. Use null if missing.")
    start: Optional[str] = Field(None, description="Start date as written, e.g. 'June 2025'. Use null if missing.")
    end: Optional[str] = Field(
        None,
        description=(
            "End date as written. If the profile shows 'Present', or the role is "
            "ongoing, return exactly the word 'Present'. Only use null when no end "
            "information appears in the text at all."
        ),
    )
    description: Optional[str] = Field(
        None,
        description=(
            "What this person did in this role, drawn from the bullet points "
            "beneath it. Keep the technologies, tools and outcomes that are named. "
            "Maximum 300 characters, so condense rather than copying every bullet "
            "in full. Null if the role has no bullets."
        ),
    )


class EducationItem(BaseModel):
    degree: Optional[str] = Field(None, description="Degree type, e.g. 'Master of Science'. Use null if missing.")
    field: Optional[str] = Field(None, description="Field of study. Use null if missing.")
    school: Optional[str] = Field(None, description="School name. Use null if missing.")
    start: Optional[str] = Field(None, description="Start date or year. Use null if missing.")
    end: Optional[str] = Field(
        None,
        description=(
            "End date or graduation year. If the programme is ongoing, return "
            "exactly 'Present'. Use null only when no end information appears."
        ),
    )


class ProfileSchema(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the person.")
    headline: Optional[str] = Field(
        None,
        description="The one line under the person's name describing what they do. Copy it as written.",
    )
    summary: Optional[str] = Field(
        None,
        description="The text under the Summary heading, copied as written. Null if there is no Summary section.",
    )
    linkedin_url: Optional[str] = Field(
        None,
        description=(
            "The LinkedIn profile URL from the Contact block at the top of the "
            "document, e.g. 'www.linkedin.com/in/sonal-naveenmeda'. Copy it exactly. "
            "Null if no LinkedIn URL appears."
        ),
    )
    education: list[EducationItem] = Field(default=[], description="List of education objects.")
    experience: list[ExperienceItem] = Field(default=[], description="List of experience objects.")
    skills: list[str] = Field(
        default=[],
        description=(
            "Every technology, tool, framework, language, platform, or professional "
            "capability named anywhere in the profile. Collect from the Top Skills "
            "sidebar, the Summary, and the experience bullet points. Include an item "
            "only if it is written in the text. Never invent a skill that would "
            "plausibly go with a job title, and never list a job title or company "
            "name as a skill."
        ),
    )
    languages: list[str] = Field(default=[], description="Spoken languages from the sidebar. Empty list if absent.")
    certifications: list[str] = Field(default=[], description="Certifications from the sidebar. Empty list if absent.")
    publications: list[str] = Field(default=[], description="Publications from the sidebar. Empty list if absent.")


# ── MODEL ─────────────────────────────────────────────────────────────────────

def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=4000,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


def is_already_processed(source_path):
    if not os.path.exists(OUTPUT_FILE):
        return False
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return any(p.get("source") == source_path for p in json.load(f))
    except json.JSONDecodeError:
        log.warning("%s is not valid JSON, treating every profile as unprocessed", OUTPUT_FILE)
        return False


# ── EXTRACTION ────────────────────────────────────────────────────────────────

def extract_profile(text, llm):
    structured_llm = llm.with_structured_output(ProfileSchema)

    prompt = f"""Extract this person's details from their LinkedIn profile export.

DOCUMENT LAYOUT
LinkedIn exports are two column, and the text extractor flattens them. A sidebar
appears FIRST, before the person's name, containing: Contact, Top Skills,
Languages, Certifications, Publications. The main content follows: name,
headline, location, Summary, Experience, Education.

PROFILE TEXT
{text}

EXTRACTION RULES

1. SKILLS COME FROM THREE PLACES. The "Top Skills" sidebar block only ever lists
   three entries, so it is never the full picture. Also collect every technology,
   tool, framework, language or platform named in the Summary and in the
   experience bullet points. Include an item only if it is written in the text.
   Do not list job titles or company names as skills.

2. LAYOUT INHERITANCE. LinkedIn groups several roles under one company heading,
   with the company named once at the top. Apply that company to every role
   listed beneath it. Do not leave company null just because it is not repeated
   on the same line as the title.

3. ONGOING ROLES. When a role shows "Present" as its end date, return exactly
   the word "Present". Do not return null: null means the date is unknown, and
   "Present" means the person is still there. These are different facts.

4. EXPERIENCE DESCRIPTIONS. Condense the bullet points under each role into that
   role's description field, keeping the technologies and outcomes named there.
   Maximum 300 characters per role.

5. LINKEDIN URL. The Contact block at the top holds a linkedin.com/in/ URL.
   Copy it into linkedin_url exactly as written.

6. COPY, DO NOT INFER. Every value must appear in the text above. If something
   is not written, leave it null rather than guessing what it probably was.
"""

    result = structured_llm.invoke(prompt)
    return result.model_dump()


def extract_with_retry(text, llm, label):
    """Retry on rate limits, which are transient and expected on the free tier."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return extract_profile(text, llm)
        except Exception as e:
            message = str(e)
            is_rate_limit = "rate_limit" in message or "429" in message

            if is_rate_limit and attempt < MAX_RETRIES:
                wait = THROTTLE_SECONDS * attempt * 2
                log.warning(
                    "Rate limited on %s (attempt %d of %d), waiting %ds",
                    label, attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            raise


def save_record(record):
    """Write after every record. A version that saved only at the end of the
    loop hit a quota limit on the last profile and lost 20 good extractions."""
    existing = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []

    existing.append(record)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    return len(existing)


def classify_failure(error):
    """Turn a provider exception into a short readable reason.

    The raw errors are long JSON blobs. The summary at the end of a run is more
    useful when it says which kind of problem each profile hit, since the fix
    differs: a rate limit just needs a rerun, while a truncated response means
    the profile is too long for the current token budget.
    """
    message = str(error)

    if "tool_use_failed" in message or "Failed to parse tool call" in message:
        return "output truncated, profile too long for max_tokens"
    if "rate_limit" in message or "429" in message:
        return "rate limited, retries exhausted"
    if "413" in message or "Request too large" in message:
        return "request too large for the tokens per minute limit"
    if "401" in message or "invalid_api_key" in message:
        return "authentication failed, check GROQ_API_KEY"
    if "timeout" in message.lower():
        return "request timed out"

    return message[:120]


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    started = time.time()

    log.info("Extraction run starting")
    log.info("Source folder : %s", DOCS_DIR)
    log.info("Output file   : %s", OUTPUT_FILE)

    llm = get_llm()
    docs = load_pdf(DOCS_DIR)
    log.info("Loaded %d documents", len(docs))

    saved_names = []
    skipped_files = []
    failed_items = []

    for i, doc in enumerate(docs, start=1):
        source_path = doc.metadata.get("source", "Unknown Source")
        label = os.path.basename(source_path)

        if is_already_processed(source_path):
            skipped_files.append(label)
            continue


        log.info("[%d/%d] Extracting %s (%d characters)", i, len(docs), label, len(doc.page_content))

        try:
            record = extract_with_retry(doc.page_content, llm, label)
            record["source"] = source_path
            total = save_record(record)
            saved_names.append(record.get("name") or label)

            log.info(
                "[%d/%d] Saved %s | %d roles, %d degrees, %d skills | file now holds %d",
                i, len(docs),
                record.get("name") or "UNNAMED",
                len(record.get("experience") or []),
                len(record.get("education") or []),
                len(record.get("skills") or []),
                total,
            )

            if not record.get("linkedin_url"):
                log.warning("No linkedin_url extracted for %s", record.get("name") or label)

        except Exception as e:
            reason = classify_failure(e)
            failed_items.append((label, reason))
            log.error("[%d/%d] Failed %s: %s", i, len(docs), label, reason)
            log.debug("Full error for %s: %s", label, e)

        time.sleep(THROTTLE_SECONDS)

    elapsed = time.time() - started

    log.info("=" * 64)
    log.info("EXTRACTION RUN SUMMARY")
    log.info("=" * 64)
    log.info("Finished in %.1fs", elapsed)
    log.info(
        "Extracted %d | Skipped %d | Failed %d",
        len(saved_names), len(skipped_files), len(failed_items),
    )

    if saved_names:
        log.info("-" * 64)
        log.info("EXTRACTED THIS RUN (%d)", len(saved_names))
        for name in saved_names:
            log.info("  ok    %s", name)

    if skipped_files:
        log.info("Skipped %d already extracted", len(skipped_files))

    if failed_items:
        log.info("-" * 64)
        log.warning("FAILED (%d)", len(failed_items))
        for f, reason in failed_items:
            log.warning("  FAIL  %-28s %s", f, reason)
        log.warning("-" * 64)
        log.warning(
            "Rerun this script to retry the failed profile(s). Profiles already "
            "in %s are skipped, so only the failures are attempted again.",
            os.path.basename(OUTPUT_FILE),
        )
    else:
        log.info("-" * 64)
        log.info("No failures.")