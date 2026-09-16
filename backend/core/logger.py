"""
One logger for the whole project.

Writes to both the console and logs/app.log, so a run that scrolls past in the
terminal is still readable afterwards. Import get_logger anywhere:

    from backend.core.logger import get_logger
    log = get_logger(__name__)
    log.info("Saved %s", name)
"""

import logging
import os
import sys

from backend.core.config import BASE_DIR

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure():
    global _configured
    if _configured:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    # INFO, not DEBUG. Libraries log heavily at DEBUG and would fill the file
    # with lines that are not about this project.
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # These libraries are noisy at INFO and drown out our own lines
    for noisy in (
        "httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers",
        "langchain", "langchain_community", "langchain_core", "langsmith",
        "openai", "groq", "transformers", "torch", "filelock", "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name):
    _configure()
    return logging.getLogger(name)