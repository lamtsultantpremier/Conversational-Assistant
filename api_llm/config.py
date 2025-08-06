import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PDF_DIR = "../documents_pdf"

PERSIST_DIR = Path().cwd() / "chroma_langchain_db"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
