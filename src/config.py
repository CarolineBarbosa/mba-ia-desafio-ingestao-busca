"""Configuração central do projeto.

Carrega as variáveis do arquivo .env (não commitado) e expõe os valores
necessários para ingestão e busca. Nenhum segredo é hardcoded aqui.

Convenção de importação: os scripts em src/ rodam com `python src/<script>.py`,
então importam os módulos vizinhos sem o prefixo `src.` (ex.: `from config import ...`).
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Google AI Studio (Gemini) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")

# --- Banco de dados local (docker compose) ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/rag",
)

# --- PDF de entrada ---
PDF_PATH = os.getenv("PDF_PATH", "document.pdf")

# Variáveis sem as quais a ingestão/busca não fazem sentido.
REQUIRED_VARS = ["GOOGLE_API_KEY"]


def missing_vars() -> list[str]:
    """Retorna as variáveis obrigatórias que não estão definidas."""
    return [name for name in REQUIRED_VARS if not os.getenv(name)]


def validate() -> None:
    """Levanta erro se alguma variável obrigatória estiver ausente."""
    missing = missing_vars()
    if missing:
        raise RuntimeError(
            "Variáveis de ambiente obrigatórias ausentes: "
            + ", ".join(missing)
            + ". Copie .env.example para .env e preencha os valores."
        )


if __name__ == "__main__":
    # Nunca imprime a chave, apenas se ela está presente.
    print("Configuração carregada:")
    print(f"  GOOGLE_API_KEY definida: {bool(GOOGLE_API_KEY)}")
    print(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"  LLM_MODEL: {LLM_MODEL}")
    print(f"  DATABASE_URL: {DATABASE_URL}")
    print(f"  PDF_PATH: {PDF_PATH}")
    print(f"  PDF existe: {os.path.isfile(PDF_PATH)}")
    print(f"  Obrigatórias ausentes: {missing_vars()}")
