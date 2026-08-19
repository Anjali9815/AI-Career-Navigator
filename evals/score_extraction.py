import os, json
from collections import defaultdict
from backend.core.config import PROFILES_PATH

def score_extraction_run():
    if not os.path.exists(PROFILES_PATH):
        print("No profiles.json found to evaluate.")
        return

    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    total_profiles = len(profiles)
    missing_names = 0
    empty_skills = 0
    missing_companies = 0
    missing_starts = 0
    missing_headlines = 0
    missing_summary = 0
    missing_languages = 0
    missing_certifications = 0
    missing_publication = 0

    missing_files = defaultdict(list)
    for p in profiles:
        if not p.get("name"):
            missing_names += 1
        if not p.get("skills"):
            empty_skills += 1
        for exp in p.get("experience", []):
            if not exp.get("company"):
                missing_companies += 1
                print(p["source"])
            if not exp.get("start"):
                missing_starts += 1
        if not p.get("headline"):
            missing_headlines += 1
            missing_files["Headline_Missing"].append(p["source"].split("/")[-1])
        if not p.get("summary"):
            missing_summary += 1
            missing_files["Summary_Missing"].append(p["source"].split("/")[-1])
        if not p.get("languages"):
            missing_languages += 1
            missing_files["Languages_Missing"].append(p["source"].split("/")[-1])
        if not p.get("certifications"):
            missing_certifications += 1
            missing_files["Certification_Missing"].append(p["source"].split("/")[-1])
        if not p.get("publications"):
            missing_publication += 1
            missing_files["Publication_Missing"].append(p["source"].split("/")[-1])


    # Aggregate metric: lower is better (0 is a perfect run)
    total_errors = missing_names + empty_skills + missing_companies + missing_starts

    print(f"=== EXTRACTION RUN REPORT ===")
    print(f"Total Profiles Evaluated : {total_profiles}")
    print(f"Missing Names            : {missing_names}")
    print(f"Empty Skills Lists       : {empty_skills}")
    print(f"Missing Companies        : {missing_companies}")
    print(f"Missing Start Dates      : {missing_starts}")
    print(f"Missing Headline         : {missing_headlines}")
    print(f"Missing Summary          : {missing_summary}")
    print(f"Missing Languages        : {missing_languages}")
    print(f"Missing Certification    : {missing_certifications}")
    print(f"Missing Publication      : {missing_publication}")
    print(f"-----------------------------")
    print(f"TOTAL RUN ERROR SCORE : {total_errors}\n")

    print(missing_files)

if __name__ == "__main__":
    score_extraction_run()
