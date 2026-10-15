from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class ConversationManager:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def add_message(self, role: str, content: str) -> None:
        cleaned = content.strip()
        if not cleaned:
            return
        self.messages.append(ChatMessage(role=role, content=cleaned))

    def clear(self) -> None:
        self.messages.clear()

    def export_to_text(self, document_name: str | None = None) -> str:
        lines = [
            "SmartDesk AI - Local File Assistant",
            f"Document: {document_name or 'No document loaded'}",
            f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
        ]

        for message in self.messages:
            label = "USER" if message.role == "user" else "SMARTDESK AI"
            lines.append(f"{label} [{message.timestamp}]")
            lines.append(message.content)
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def save_conversation(self, file_path: str | Path, document_name: str | None = None) -> str:
        destination = Path(file_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = self.export_to_text(document_name=document_name)
        destination.write_text(text, encoding="utf-8")
        return str(destination)
