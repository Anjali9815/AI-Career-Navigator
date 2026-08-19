import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
PROFILES_PATH = os.path.join(BASE_DIR, "data", "profiles.json")
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
COLLECTION_NAME = "people"
DATA_DIR = os.path.join(BASE_DIR, "data")

JSON_DIR = os.path.join(BASE_DIR, "data", "retrieval_tests.json")
DATASET_NAME = "AI Career Navigator Evaluation"
RELEVANCE_THRESHOLD = -8.10
DATASET_NAME = "AI Career Navigator Evaluation"
# Make sure the 'data' directory exists before processing
os.makedirs(DATA_DIR, exist_ok=True)

DISTANCE_THRESHOLD = -8