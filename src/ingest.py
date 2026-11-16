"""Script de ingestão do PDF.

Etapa 4 (atual): validação do caminho/arquivo e leitura do PDF.
Sem embeddings e sem banco de dados ainda.
"""
import os

from langchain_community.document_loaders import PyPDFLoader

from config import PDF_PATH


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


def ingest() -> list:
    """Orquestra a ingestão: valida o PDF e então o lê.

    Nas próximas etapas, esta função também vai fatiar em chunks,
    gerar embeddings e persistir no pgVector.
    """
    validate_pdf_path(PDF_PATH)
    return load_pdf(PDF_PATH)


if __name__ == "__main__":
    ingest()
