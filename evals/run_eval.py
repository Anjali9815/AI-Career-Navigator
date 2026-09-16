from collections import defaultdict
import json
import os
from datetime import datetime
from langsmith import Client, evaluate
from backend.src.vectorstore import final_production_search
from backend.core.config import BASE_DIR, JSON_DIR, DATASET_NAME, DISTANCE_THRESHOLD
from dotenv import load_dotenv
load_dotenv()

from backend.core.logger import get_logger
log = get_logger("run_eval")

client = Client()

def load_json_file(file_path):
    with open(file_path, "r") as f:
        query_data = json.load(f)
        log.info("Loaded query data: %d items", len(query_data))

    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Profile retrieval validation tests.")

        examples = []

        for item in query_data:
            examples.append(
            {
                "inputs": {"query": item["query"]},
                "outputs": {
                    "should_match": item["should_match"],
                    "expected_names": item.get("expected_names", []),
                },
            })
        client.create_examples(dataset_id=dataset.id, examples = examples)
        log.info("Successfully created data in Langsmith")
        pass

def run_vector_search(inputs : dict)-> dict:
    query = inputs["query"]

    results = final_production_search(query, top_k=1)
    if results:
        doc, score  = results[0]
        return {"matched_name" : doc.metadata.get("name", "Unknown"), "score" : float(score)}
    return {"matched_name" : "Unknown", "Score" : -15.0}


def profile_match_grader(root_run, example):
    predicted_name = root_run.outputs.get("matched_name")
    search_score = root_run.outputs.get("score")

    should_match = example.outputs.get("should_match")
    expected_names = example.outputs.get("expected_names", [])

    if not should_match:
        if search_score < DISTANCE_THRESHOLD or predicted_name == "Unknown":
            return {"key": "correct_retrieval", "score": 1}
        return {"key": "correct_retrieval", "score": 0}

    if should_match:
        if predicted_name in expected_names and search_score >= DISTANCE_THRESHOLD:
            return {"key": "correct_retrieval", "score": 1}
        return {"key": "correct_retrieval", "score": 0}
    return {"key": "correct_retrieval", "score": 0}


if __name__ == "__main__":
    load_json_file(JSON_DIR)
    experiment_result = evaluate(
        run_vector_search,
        data = DATASET_NAME,
        evaluators=[profile_match_grader],
        experiment_prefix="Hybrid-Rerank-Production-v1"
    )
    log.info("Evaluation execution complete.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    df = experiment_result.to_pandas()
    out = os.path.join(BASE_DIR, "evals", "results", f"retrieval_{stamp}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    score = df["feedback.correct_retrieval"].mean()
    log.info("correct_retrieval: %.3f over %d cases", score, len(df))
    log.info("Saved results to %s", out)

