from backend.src.vectorstore import final_production_search


if __name__ == "__main__":
    query = "Who knows chinese?"
    results_score = final_production_search(query, k=3)
    for doc, score in results_score:
        print(score, doc.metadata["name"])
    
