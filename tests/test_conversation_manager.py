import tempfile
import unittest
from pathlib import Path

from src.conversation_manager import ConversationManager


class ConversationManagerTests(unittest.TestCase):
    def test_add_message_and_export(self) -> None:
        manager = ConversationManager()
        manager.add_message("user", "What are the key findings?")
        manager.add_message("assistant", "The report highlights rising costs and lower productivity.")

        exported = manager.export_to_text("Example.pdf")
        self.assertIn("USER", exported)
        self.assertIn("SMARTDESK AI", exported)
        self.assertIn("key findings", exported.lower())

    def test_save_conversation_creates_file(self) -> None:
        manager = ConversationManager()
        manager.add_message("user", "Summarize this report.")
        manager.add_message("assistant", "This is a summary of the report.")

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "conversation.txt"
            saved_path = manager.save_conversation(file_path, document_name="report.pdf")
            self.assertTrue(Path(saved_path).exists())
            text = Path(saved_path).read_text(encoding="utf-8")
            self.assertIn("report.pdf", text)
            self.assertIn("SUMMARIZE THIS REPORT", text.upper())


if __name__ == "__main__":
    unittest.main()
