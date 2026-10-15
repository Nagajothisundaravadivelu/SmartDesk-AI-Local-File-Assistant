import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from src.document_qa import split_into_chunks
from src.document_reader import DocumentError, load_document
from src.ollama_client import OllamaClient, OllamaError


def create_minimal_pdf(text: str = "Quarterly report summary") -> bytes:
    objects: list[str] = []
    stream = f"BT /F1 12 Tf 50 50 Td ({text}) Tj ET"
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 150] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")
    objects.append(f"<< /Length {len(stream) + 5} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf: list[bytes] = [b"%PDF-1.4\n"]
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(b"".join(pdf)))
        pdf.append(f"{idx} 0 obj\n{obj}\nendobj\n".encode("latin-1"))

    xref_offset = len(b"".join(pdf))
    pdf.append(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets[1:]:
        pdf.append(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.append(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("latin-1"))
    return b"".join(pdf)


class DocumentReaderTests(unittest.TestCase):
    def test_load_txt_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("Project overview\n\nThis file has meeting notes.", encoding="utf-8")
            result = load_document(str(path))
            self.assertEqual("Project overview\n\nThis file has meeting notes.", result.text)
            self.assertEqual(len(result.text), result.char_count)

    def test_empty_text_file_raises_document_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.txt"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(DocumentError):
                load_document(str(path))

    def test_missing_document_raises_document_error(self) -> None:
        with self.assertRaises(DocumentError):
            load_document("C:/does/not/exist.txt")

    def test_extract_pdf_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.pdf"
            path.write_bytes(create_minimal_pdf("Quarterly report summary"))
            result = load_document(str(path))
            self.assertIn("Quarterly report summary", result.text)
            self.assertEqual(result.extension, ".pdf")

    def test_unsupported_file_raises_document_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.csv"
            path.write_text("name,value\nalpha,1", encoding="utf-8")
            with self.assertRaises(DocumentError):
                load_document(str(path))

    def test_chunking_large_document(self) -> None:
        text = "\n\n".join([f"Section {index}: " + ("This is a detailed paragraph about the project and its goals. " * 15) for index in range(1, 10)])
        chunks = split_into_chunks(text, chunk_size=600)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 600 for chunk in chunks))

    def test_ollama_connection_failure(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", model="llama3.2")
        with patch("src.ollama_client.requests.get", side_effect=requests.exceptions.ConnectionError("down")):
            self.assertFalse(client.check_connection())

    def test_ollama_model_missing(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", model="llama3.2")
        fake_response = Mock()
        fake_response.status_code = 200
        fake_response.text = ""
        fake_response.json.return_value = {"models": [{"name": "mistral:latest"}]}

        with patch("src.ollama_client.requests.get", return_value=fake_response):
            self.assertFalse(client.check_model_available())

    def test_ollama_chat_reports_missing_model(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", model="llama3.2")
        fake_response = Mock()
        fake_response.status_code = 404
        fake_response.text = '{"error":"model \\"llama3.2\\" not found, try pulling it first"}'

        with patch("src.ollama_client.requests.post", return_value=fake_response):
            with self.assertRaisesRegex(OllamaError, "ollama pull llama3.2"):
                client.generate_chat_response("Summarize this document.")


if __name__ == "__main__":
    unittest.main()
