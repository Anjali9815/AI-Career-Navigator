import os
import json
from typing import Optional
from dotenv import load_dotenv
# 1. Swapped the import to Groq
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from backend.src.loader import load_pdf

from backend.core.config import DOCS_DIR, PROFILES_PATH
OUTPUT_FILE = PROFILES_PATH
load_dotenv()

# Structure for Experience Objects
class ExperienceItem(BaseModel):
    title: Optional[str] = Field(None, description="Job title. Use null if missing.")
    company: Optional[str] = Field(None, description="Company name. Use null if missing.")
    start: Optional[str] = Field(None, description="Start date. Use null if missing.")
    end: Optional[str] = Field(None, description="End date. Use 'Present' if current role. Use null if missing.")

# Structure for Education Objects
class EducationItem(BaseModel):
    degree: Optional[str] = Field(None, description="Degree type (e.g. B.S.). Use null if missing.")
    field: Optional[str] = Field(None, description="Field of study. Use null if missing.")
    school: Optional[str] = Field(None, description="School name. Use null if missing.")
    start: Optional[str] = Field(None, description="Start date. Use null if missing.")
    end: Optional[str] = Field(None, description="End date or graduation year. Use null if missing.")

# Main Profile Structure
class ProfileSchema(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the person.")
    education: list[EducationItem] = Field(default=[], description="List of education objects.")
    experience: list[ExperienceItem] = Field(default=[], description="List of experience objects.")
    skills: list[str] = Field(
        default=[], 
        description="List of clean technical/professional skills. Remove generic noise like 'Grading' or 'Lesson Planning'."
    )
    headline: Optional[str]
    summary: Optional[str]
    languages: list[str] = []
    certifications: list[str] = []
    publications: list[str] = []

# 2. Updated to use ChatGroq with a powerful extraction model (Llama-3.3-70b)
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b", # Ultra-fast and highly precise for extraction tasks
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

def is_already_processed(source_path):
    if not os.path.exists(OUTPUT_FILE):
        return False
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return any(p.get("source") == source_path for p in json.load(f))
    except json.JSONDecodeError:
        return False



def extract_profile(text, llm):
    # This instructs Groq to output structural JSON directly mapping our schema
    structured_llm = llm.with_structured_output(ProfileSchema)
    
    prompt = f"""
        Extract this person's details from their LinkedIn profile export.

        Document layout: a sidebar appears first (Contact, Top Skills,
        Languages, Certifications, Publications), followed by the main
        content (name, headline, location, Summary, Experience, Education).

        headline: the one line under the person's name describing what
        they do. Copy it as written.

        summary: the text under the Summary heading, copied as written.
        Null if there is no Summary section.

        skills: collect from three places, not just Top Skills, which
        only ever lists three.
        1. The Top Skills sidebar section
        2. Technologies named in the Summary
        3. Technologies named in the experience bullet points
        Include a technology only if it is named in the text. Do not
        invent skills that would plausibly go with a job title.

        languages, certifications, publications: copy the entries from
        those sidebar sections. Empty list if the section is absent.

        experience description: the bullet points under each role,
        copied as written. Null if the role has no bullets.


        Layout Inheritance Rule:
        - If multiple job titles/roles are listed sequentially under a single parent company heading, apply that same company name to all of those roles. Do not leave the company field null just because it isn't explicitly repeated on the same line as the title.

        Profile Text:
        {text}
    """

    
    result = structured_llm.invoke(prompt)
    return result.model_dump()


if __name__ == "__main__":
    llm = get_llm()
    
    docs = load_pdf(DOCS_DIR)
    profiles = []
    for doc in docs:
        try:
            source_path = doc.metadata.get("source", "Unknown Source")
            if is_already_processed(source_path):
                continue
            # 1. Extract the structured dictionary profile from the text
            record = extract_profile(doc.page_content, llm)
            
            # 2. Append the file source metadata
            record["source"] = doc.metadata.get("source", "Unknown Source")
            
            # --- Incremental Save Logic ---
            existing_profiles = []
            
            # Read whatever has already been successfully written to disk
            if os.path.exists(OUTPUT_FILE):
                try:
                    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                        existing_profiles = json.load(f)
                except json.JSONDecodeError:
                    # Handle edge case if file exists but is corrupted/empty
                    existing_profiles = []

            # Add the freshly extracted record to our local list
            existing_profiles.append(record)

            # Rewrite the updated file immediately to preserve your API investment
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(existing_profiles, f, indent=2, ensure_ascii=False)
                
            print(f"Successfully processed and saved: {record['source']}")

        except Exception as e:
            source_info = doc.metadata.get("source", "Unknown Source")
            print("Failed:", source_info, e)

    print("All iterations finished!")
