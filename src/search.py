"""Semantic search: retrieve chunks and answer with the LLM."""
import sys

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres import PGVector

from config import DATABASE_URL, LLM_MODEL
from ingest import COLLECTION_NAME, get_embeddings

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

# How many most-similar chunks to return.
TOP_K = 10

# Fallback answer when there is not enough context.
# Kept in sync with the fallback text in PROMPT_TEMPLATE.
FALLBACK_ANSWER = "Não tenho informações necessárias para responder sua pergunta."


def build_vector_store() -> PGVector:
    """Connect to pgVector with the same collection and model as ingestion."""
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def retrieve(question: str, k: int = TOP_K) -> list:
    """Return the `k` chunks most similar to the question and their scores.

    Returns a list of `(Document, score)`. The score is a distance: the
    LOWER it is, the more similar (0 = identical).
    """
    return build_vector_store().similarity_search_with_score(question, k=k)


def show_results(question: str, results: list) -> None:
    """Print the question, the retrieved chunks and each score."""
    print(f"Pergunta: {question}")
    print(f"Chunks recuperados: {len(results)}")
    print()
    for i, (doc, score) in enumerate(results, start=1):
        print(f"--- Resultado {i} (score: {score:.4f}) ---")
        print(doc.page_content[:300])
        print()


def get_llm() -> ChatGoogleGenerativeAI:
    """Return the chat model configured in `LLM_MODEL`."""
    return ChatGoogleGenerativeAI(model=LLM_MODEL)


def format_context(results: list) -> str:
    """Join the retrieved chunks into a single text for the prompt."""
    return "\n\n".join(doc.page_content for doc, _ in results)


def search_prompt(question: str) -> str:
    """RAG pipeline: retrieve context, build the prompt and answer with the LLM.

    If nothing is retrieved, returns `FALLBACK_ANSWER` without calling the model.
    """
    results = retrieve(question)
    if not results:
        return FALLBACK_ANSWER

    prompt = PROMPT_TEMPLATE.format(
        contexto=format_context(results),
        pergunta=question,
    )
    return get_llm().invoke(prompt).content


if __name__ == "__main__":
    pergunta = sys.argv[1] if len(sys.argv) > 1 else "Qual é o assunto principal do documento?"
    print(search_prompt(pergunta))
