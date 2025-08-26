import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PDF_DIR = Path.cwd() / "documents_pdf"
PERSIST_DIR = Path().cwd() / "final_chromadb_database"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
