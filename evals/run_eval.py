from collections import defaultdict
import json
import os
from langsmith import Client, evaluate
from backend.src.vectorstore import final_production_search
from backend.core.config import JSON_DIR, DATASET_NAME, DISTANCE_THRESHOLD

client = Client()

def load_json_file(file_path):
    with open(file_path, "r") as f:
        query_data = json.load(f)
        print(query_data)

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
        print("Successfully data created in Langsmith")
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
    print("Evaluation execution complete.")

