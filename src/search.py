"""Busca semântica: recupera chunks e responde com o LLM."""
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

# Quantos chunks mais semelhantes retornar na busca.
TOP_K = 10

# Resposta de fallback quando não há contexto suficiente.
# Mantida em sincronia com o texto de fallback do PROMPT_TEMPLATE.
FALLBACK_ANSWER = "Não tenho informações necessárias para responder sua pergunta."


def build_vector_store() -> PGVector:
    """Conecta ao pgVector com a mesma coleção e modelo da ingestão."""
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def retrieve(question: str, k: int = TOP_K) -> list:
    """Retorna os `k` chunks mais semelhantes à pergunta e suas pontuações.

    Devolve uma lista de `(Document, score)`. O score é uma distância:
    quanto MENOR, mais semelhante (0 = idêntico).
    """
    return build_vector_store().similarity_search_with_score(question, k=k)


def show_results(question: str, results: list) -> None:
    """Imprime a pergunta, os chunks recuperados e o score de cada um."""
    print(f"Pergunta: {question}")
    print(f"Chunks recuperados: {len(results)}")
    print()
    for i, (doc, score) in enumerate(results, start=1):
        print(f"--- Resultado {i} (score: {score:.4f}) ---")
        print(doc.page_content[:300])
        print()


def get_llm() -> ChatGoogleGenerativeAI:
    """Retorna o modelo de chat configurado em `LLM_MODEL`."""
    return ChatGoogleGenerativeAI(model=LLM_MODEL)


def format_context(results: list) -> str:
    """Junta os chunks recuperados em um único texto para o prompt."""
    return "\n\n".join(doc.page_content for doc, _ in results)


def search_prompt(question: str) -> str:
    """Pipeline RAG: recupera contexto, monta o prompt e responde com o LLM.

    Se nada for recuperado, devolve `FALLBACK_ANSWER` sem chamar o modelo.
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
