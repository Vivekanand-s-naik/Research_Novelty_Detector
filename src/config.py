"""
Central configuration.

API keys are read from environment variables (loaded from a local `.env`
file).  Nothing secret is ever hardcoded here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root (does not override real env variables).
load_dotenv()

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_papers"
RETRIEVED_DIR = DATA_DIR / "retrieved_papers"
SEARCH_CACHE_DIR = RETRIEVED_DIR / "search_cache"
CONTENT_CACHE_DIR = RETRIEVED_DIR / "content_cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Models / limits (safe defaults that keep API usage low)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

MAX_QUERIES = 3            # Tavily calls per analysis (1 credit each, basic)
RESULTS_PER_QUERY = 5      # Results requested per Tavily query
TOP_K_COMPARE = 3          # Papers sent to Groq for the final comparison


class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing."""


PLACEHOLDER_PREFIXES = ("your_", "<your", "replace_", "enter_", "xxx")


def _is_placeholder(value):
    if not value:
        return True
    lower = value.strip().lower()
    return (
        lower in ("none", "null", "todo", "changeme")
        or lower.startswith(PLACEHOLDER_PREFIXES)
        or lower.endswith("_here")
    )


def get_env(name, default=None):
    """Return a stripped environment variable or `default`."""
    value = os.getenv(name, "")
    value = value.strip() if value else ""
    if _is_placeholder(value):
        return default
    return value or default


def require_env(name):
    """Return a required environment variable or raise ConfigError."""
    raw = (os.getenv(name, "") or "").strip()
    if not raw or _is_placeholder(raw):
        hint = (
            f" (found placeholder '{raw}')" if raw else ""
        )
        raise ConfigError(
            f"Missing or placeholder environment variable '{name}'{hint}. "
            f"Set a valid key in {PROJECT_ROOT / '.env'} "
            f"(see .env.example) or via the Streamlit sidebar."
        )
    return raw


def get_groq_model():
    return get_env("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def api_key_status():
    """Return {key_name: bool} without revealing the key values."""
    return {
        "TAVILY_API_KEY": bool(get_env("TAVILY_API_KEY")),
        "GROQ_API_KEY": bool(get_env("GROQ_API_KEY")),
    }


def ensure_directories():
    for directory in (UPLOAD_DIR, RETRIEVED_DIR, SEARCH_CACHE_DIR,
                      CONTENT_CACHE_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

