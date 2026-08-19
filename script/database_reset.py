import os
import shutil
from langchain_chroma import Chroma
from backend.src.embedder import get_embedder
from backend.core.config import CHROMA_DIR

if __name__ == "__main__":
    print("Force resetting Chroma DB...")

    if os.path.exists(CHROMA_DIR):
        try:
            # 1. Connect to the existing database
            db = Chroma(
                collection_name="people",
                persist_directory=CHROMA_DIR,
                embedding_function=get_embedder(),
            )
            # 2. Command Chroma to explicitly clear the collections
            db.delete_collection()
            print("Collection deleted from Chroma cache.")
        except Exception as e:
            print(f"Note: Cache clear skipped ({e})")

        # 3. Physically delete the underlying SQLite/Parquet files
        shutil.rmtree(CHROMA_DIR)
        print("Database folder physically deleted.")

    print("Hard reset complete! Your vector environment is now 100% clean.")
