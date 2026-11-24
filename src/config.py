"""Central project configuration.

Loads variables from the .env file (not committed) and exposes the values
needed for ingestion and search. No secrets are hardcoded here.

Import convention: scripts under src/ run as `python src/<script>.py`, so they
import sibling modules without the `src.` prefix (e.g. `from config import ...`).
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Google AI Studio (Gemini) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")

# --- Local database (docker compose) ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/rag",
)

# --- Input PDF ---
PDF_PATH = os.getenv("PDF_PATH", "document.pdf")

# Variables without which ingestion/search make no sense.
REQUIRED_VARS = ["GOOGLE_API_KEY"]


def missing_vars() -> list[str]:
    """Return the required variables that are not set."""
    return [name for name in REQUIRED_VARS if not os.getenv(name)]


def validate() -> None:
    """Raise an error if any required variable is missing."""
    missing = missing_vars()
    if missing:
        raise RuntimeError(
            "Variáveis de ambiente obrigatórias ausentes: "
            + ", ".join(missing)
            + ". Copie .env.example para .env e preencha os valores."
        )


if __name__ == "__main__":
    # Never prints the key, only whether it is set.
    print("Configuração carregada:")
    print(f"  GOOGLE_API_KEY definida: {bool(GOOGLE_API_KEY)}")
    print(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"  LLM_MODEL: {LLM_MODEL}")
    print(f"  DATABASE_URL: {DATABASE_URL}")
    print(f"  PDF_PATH: {PDF_PATH}")
    print(f"  PDF existe: {os.path.isfile(PDF_PATH)}")
    print(f"  Obrigatórias ausentes: {missing_vars()}")
