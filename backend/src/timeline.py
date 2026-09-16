"""
Builds a person's career timeline in code rather than asking the LLM for it.

Why this exists: the model was returning career steps out of order and
occasionally dropping a degree. The dates are already in profiles.json, so
ordering is a sorting problem, not a language problem. Sorting in code removes
a whole class of hallucination.
"""

import json
from backend.core.config import PROFILES_PATH

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10,
    "nov": 11, "dec": 12,
}

# Entries with no start date sort to the END, not the start. A degree with a
# missing date is not the earliest thing that happened, it is simply unknown,
# and putting it first misleads the reader.
MISSING = (9998, 12)
# Current roles sort last of all
PRESENT = (9999, 12)


def date_key(value):
    """
    Turn a date string into a sortable (year, month) tuple.

    Handles the four shapes that appear in profiles.json:
        "August 2022"  → (2022, 8)
        "2015"         → (2015, 1)
        "2023-12"      → (2023, 12)
        "Present"      → (9999, 12)
        None           → (9998, 12)   sorts to the end, date unknown
    """
    if not value:
        return MISSING

    text = str(value).strip()
    if not text:
        return MISSING

    if text.lower() in {"present", "current", "now"}:
        return PRESENT

    # ISO style: 2023-12 or 2023-12-01
    if "-" in text:
        parts = text.split("-")
        try:
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 else 1
            return (year, month)
        except (ValueError, IndexError):
            pass

    # "August 2022" or "Aug 2022"
    tokens = text.replace(",", " ").split()
    year = None
    month = 1

    for token in tokens:
        lowered = token.lower()
        if lowered in MONTHS:
            month = MONTHS[lowered]
        elif token.isdigit() and len(token) == 4:
            year = int(token)

    if year is None:
        return MISSING

    return (year, month)


def format_range(start, end):
    """Render a date range for display. Returns an empty string when unknown."""
    if not start and not end:
        return ""
    if start and end:
        return f"{start} to {end}"
    return start or end


def build_timeline(record):
    """
    Merge one person's education and experience into a single list of steps,
    sorted oldest first.

    Each step:
        {
          "kind":  "education" | "experience",
          "label": "BTech Electronics and Communication Engineering, GRIET",
          "dates": "May 2018 to April 2022",
          "sort":  (2018, 5)          # dropped before returning
        }
    """
    steps = []

    for edu in record.get("education") or []:
        degree = edu.get("degree")
        field = edu.get("field")
        school = edu.get("school")

        if degree and field:
            label = f"{degree} in {field}"
        else:
            label = degree or field or "Studied"

        if school:
            label = f"{label}, {school}"

        steps.append({
            "kind": "education",
            "label": label,
            "dates": format_range(edu.get("start"), edu.get("end")),
            "sort": date_key(edu.get("start")),
        })

    for exp in record.get("experience") or []:
        title = exp.get("title")
        company = exp.get("company")

        if title and company:
            label = f"{title} at {company}"
        else:
            label = title or company or "Role"

        steps.append({
            "kind": "experience",
            "label": label,
            "dates": format_range(exp.get("start"), exp.get("end")),
            "sort": date_key(exp.get("start")),
        })

    steps.sort(key=lambda s: s["sort"])

    for s in steps:
        del s["sort"]

    return steps


# ── lookup ────────────────────────────────────────────────────────────────────
# Loaded once at import. The profile set only changes when ingestion reruns.

def _load_profiles():
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _index_by_name(profiles):
    index = {}
    for p in profiles:
        name = p.get("name")
        if name:
            index[name.strip().lower()] = p
    return index


_PROFILES = _load_profiles()
_BY_NAME = _index_by_name(_PROFILES)


def timeline_for(name):
    """
    Look up a person by name and return their sorted timeline.
    Returns an empty list when the name is not found, so a mismatch
    degrades quietly instead of raising.
    """
    if not name:
        return []

    record = _BY_NAME.get(name.strip().lower())
    if not record:
        return []

    return build_timeline(record)


def profile_for(name):
    """Return the whole record for a name, or None."""
    if not name:
        return None
    return _BY_NAME.get(name.strip().lower())


if __name__ == "__main__":
    # Date parser checks
    cases = ["August 2022", "2015", "2023-12", "Present", None, "", "Jan 2020"]
    print("date_key:")
    for c in cases:
        print(f"  {str(c):<16} → {date_key(c)}")

    # Timeline for every profile
    print("\ntimelines:")
    for p in _PROFILES:
        print(f"\n{p.get('name')}")
        for step in build_timeline(p):
            marker = "EDU" if step["kind"] == "education" else "JOB"
            dates = f"  ({step['dates']})" if step["dates"] else ""
            print(f"  [{marker}] {step['label']}{dates}")