"""Script de ingestão do PDF.

"""
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATABASE_URL, EMBEDDING_MODEL, PDF_PATH

# Parâmetros de fatiamento (requisito do desafio).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Nome da coleção no pgVector (isola os documentos deste RAG).
COLLECTION_NAME = "rag_documentos"


def load_pdf(pdf_path: str) -> list:
    """Lê o PDF em `pdf_path` e imprime logs (páginas, caracteres, amostra).

    Devolve uma lista de `Document`, um por página.
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(f"Arquivo lido: {pdf_path}")
    print(f"Total de páginas (documentos): {len(docs)}")
    print(f"Total de caracteres extraídos: {sum(len(d.page_content) for d in docs)}")
    print()

    if not docs:
        print("Aviso: nenhum documento foi carregado.")
    else:
        amostra = docs[0].page_content
        print(
            f"--- Amostra da página 1 "
            f"({len(amostra)} caracteres, exibindo até 500) ---"
        )
        print(amostra[:500])

    return docs


def validate_pdf_path(pdf_path: str) -> None:
    """Valida o caminho do PDF: definido, existente e sendo um arquivo."""
    if not pdf_path:
        raise ValueError("PDF_PATH não está definido. Verifique o arquivo .env.")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"O caminho informado não existe: {pdf_path}")

    if not os.path.isfile(pdf_path):
        raise ValueError(f"O caminho informado não é um arquivo: {pdf_path}")


def split_documents(docs: list) -> list:
    """Fatia os documentos em chunks (CHUNK_SIZE caracteres, CHUNK_OVERLAP de overlap).

    Devolve uma lista de `Document` menores, mantendo os metadados originais.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    tamanhos = [len(c.page_content) for c in chunks]
    print(f"Total de chunks gerados: {len(chunks)}")
    if tamanhos:
        print(
            f"Tamanho dos chunks — mínimo: {min(tamanhos)}, "
            f"máximo: {max(tamanhos)}, média: {sum(tamanhos) // len(tamanhos)}"
        )
    print("\n")

    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i + 1} ({len(chunk.page_content)} caracteres) ---")
        print(chunk.page_content[:200])
        print("\n")

    return chunks


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Retorna o modelo de embeddings configurado em `EMBEDDING_MODEL`."""
    print(f"Modelo de embeddings configurado: {EMBEDDING_MODEL}")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def persist_chunks(chunks: list) -> None:
    """Gera os embeddings dos chunks e persiste no pgVector.

    `pre_delete_collection=True` recria a coleção do zero, evitando
    duplicatas quando a ingestão roda mais de uma vez.
    """
    PGVector.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        pre_delete_collection=True,
    )
    print(f"Chunks persistidos no pgVector (coleção '{COLLECTION_NAME}'): {len(chunks)}")


def ingest() -> list:
    """Orquestra a ingestão: valida, lê, fatia e persiste no pgVector."""
    validate_pdf_path(PDF_PATH)
    docs = load_pdf(PDF_PATH)
    chunks = split_documents(docs)
    persist_chunks(chunks)
    return chunks


if __name__ == "__main__":
    ingest()
