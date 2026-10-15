from __future__ import annotations

from .document_qa import build_summary_prompt, split_into_chunks
from .ollama_client import OllamaClient


def summarize_document(document_text: str, ollama_client: OllamaClient) -> str:
    if not document_text or not document_text.strip():
        raise ValueError("The document is empty and cannot be summarized.")

    chunks = split_into_chunks(document_text, chunk_size=2200)
    if len(chunks) == 1:
        prompt = build_summary_prompt(document_text)
        system_prompt = (
            "You are a careful document summarizer. Extract only real information from the provided document "
            "and organize it with clear headings."
        )
        return ollama_client.generate_chat_response(prompt, system_prompt=system_prompt)

    partial_summaries: list[str] = []
    for index, chunk in enumerate(chunks[:8], start=1):
        system_prompt = "You are summarizing a chunk of a larger document. Preserve facts and figures exactly."
        prompt = (
            "Summarize this fragment of the document in a few sentences, preserving key facts and figures. "
            f"Chunk {index}:\n{chunk}"
        )
        chunk_summary = ollama_client.generate_chat_response(prompt, system_prompt=system_prompt)
        partial_summaries.append(f"Chunk {index}: {chunk_summary.strip()}")

    combined_prompt = (
        "Combine the following chunk summaries into one coherent final summary for the full document. "
        "Use the headings Overview, Main points, Key findings, Important dates, names, or figures, "
        "Risks or challenges, and Conclusion. Keep details factual and avoid fabricating missing information.\n\n"
        + "\n\n".join(partial_summaries)
    )
    system_prompt = "You are combining multiple partial document summaries into a final structured summary."
    return ollama_client.generate_chat_response(combined_prompt, system_prompt=system_prompt)
