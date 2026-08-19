# profile_text.py

def profile_to_text(record):
    """Encodes all details into clean structured layout blocks."""
    name = record.get("name") or "Unknown"
    headline = record.get("headline") or "Unknown"
    summary = record.get("summary") or "Unknown"

    skills = ", ".join(record.get("skills") or [])
    languages = ", ".join(record.get("languages") or [])
    certifications = ", ".join(record.get("certifications") or [])

    experience_items = []
    for exp in record.get("experience") or []:
        company = exp.get("company") or "Unknown"
        title = exp.get("title") or "Unknown"
        experience_items.append(f"{title} at {company}")
    experience_str = " | ".join(experience_items)

    education_items = []
    for edu in record.get("education") or []:
        degree = edu.get("degree") or "Unknown"
        field = edu.get("field") or "Unknown"
        education_items.append(f"{degree} in {field}")
    education_str = " | ".join(education_items)

    return (
        f"CANDIDATE NAME: {name}\n"
        f"PROFESSIONAL HEADLINE: {headline}\n"
        f"CORE SKILLS: {skills}\n"
        f"LANGUAGES: {languages}\n"
        f"CERTIFICATIONS: {certifications}\n"
        f"PROFESSIONAL SUMMARY: {summary}\n"
        f"WORK HISTORY: {experience_str}\n"
        f"EDUCATION BACKGROUND: {education_str}"
    )
