"""Testes unitários (offline) das partes determinísticas do RAG.

Não exigem rede nem banco: cobrem a validação de configuração, a validação
do caminho do PDF, o fatiamento de chunks e a formatação de contexto.

Rodar com:
    venv/bin/python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.documents import Document  # noqa: E402

import config  # noqa: E402
from ingest import split_documents, validate_pdf_path  # noqa: E402
from search import format_context  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_validate_raises_without_required_vars(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                config.validate()

    def test_missing_vars_reports_google_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIn("GOOGLE_API_KEY", config.missing_vars())


class ValidatePdfPathTests(unittest.TestCase):
    def test_empty_path_raises_value_error(self):
        with self.assertRaises(ValueError):
            validate_pdf_path("")

    def test_nonexistent_path_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            validate_pdf_path("/caminho/que/nao/existe.pdf")

    def test_directory_path_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                validate_pdf_path(tmpdir)

    def test_existing_file_does_not_raise(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            self.assertIsNone(validate_pdf_path(f.name))


class SplitDocumentsTests(unittest.TestCase):
    def test_long_text_splits_into_multiple_chunks(self):
        docs = [Document(page_content="palavra " * 2000, metadata={})]
        chunks = split_documents(docs)
        self.assertGreater(len(chunks), 1)

    def test_short_text_stays_single_chunk(self):
        docs = [Document(page_content="texto curto", metadata={})]
        chunks = split_documents(docs)
        self.assertEqual(len(chunks), 1)


class FormatContextTests(unittest.TestCase):
    def test_joins_chunks_with_blank_line(self):
        results = [
            (Document(page_content="primeiro"), 0.1),
            (Document(page_content="segundo"), 0.3),
        ]
        self.assertEqual(format_context(results), "primeiro\n\nsegundo")


if __name__ == "__main__":
    unittest.main()
