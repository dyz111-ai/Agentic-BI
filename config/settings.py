import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

DB_URL = os.getenv("DB_URL", f"sqlite:///{BASE_DIR / 'data' / 'sample_olist.db'}")
AUTO_REFRESH_MV = os.getenv("AUTO_REFRESH_MV", "1") == "1"

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

OUTPUT_DIR = BASE_DIR / "outputs" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
