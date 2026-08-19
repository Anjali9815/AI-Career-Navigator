import json
import os
import warnings
from backend.src.vectorstore import final_production_search
from backend.core.config import JSON_DIR
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")


if __name__ == "__main__":
    if not os.path.exists(JSON_DIR):
        print(f"Error: Could not locate test file at {JSON_DIR}")
        exit()

    with open(JSON_DIR, "r", encoding="utf-8") as f:
        query_data = json.load(f)
    for item in query_data:
        print(f"\nQUERY: {item['query']} (Should Match: {item['should_match']})")
        print(f"{'RANK':<5} | {'RERANK SCORE':<12} | {'MATCHED NAME':<22}")

        results = final_production_search(item["query"], top_k=3)
        if results:
            for rank, (doc, score) in enumerate(results, start=1):
                name = doc.metadata.get("name", "Unknown")
                print(f"{rank:<4} | {round(score, 4):<12} | {name:<22}")
        else:
            print("No profiles returned from production engine.")


