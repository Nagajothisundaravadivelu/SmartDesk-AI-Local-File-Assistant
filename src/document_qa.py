from __future__ import annotations

import re
from collections import Counter

from .config import MAX_CHUNK_SIZE, MAX_CONTEXT_CHARS, MAX_RETRIEVED_CHUNKS


def normalize_text(text: str) -> str:
    return re.sub(r"\r\n?", "\n", text).replace("\r", "\n").strip()


def split_into_chunks(text: str, chunk_size: int = MAX_CHUNK_SIZE) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", normalized) if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(paragraph) <= chunk_size:
                current = paragraph
            else:
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                segment = ""
                for sentence in sentences:
                    if not sentence:
                        continue
                    pending = f"{segment} {sentence}" if segment else sentence
                    if len(pending) <= chunk_size:
                        segment = pending
                    else:
                        if segment:
                            chunks.append(segment)
                        segment = sentence
                if segment:
                    current = segment
    if current:
        chunks.append(current)

    if not chunks:
        return [normalized[:chunk_size]]
    return chunks


def retrieve_relevant_chunks(text: str, question: str, chunk_size: int = MAX_CHUNK_SIZE, max_chunks: int = MAX_RETRIEVED_CHUNKS) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks = split_into_chunks(normalized, chunk_size=chunk_size)
    if len(chunks) <= 1:
        return chunks

    query_words = [word.lower() for word in re.findall(r"[A-Za-z0-9]+", question) if word]
    if not query_words:
        return chunks[:max_chunks]

    word_counts = Counter(query_words)
    scored: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(chunk_lower.count(word) * frequency for word, frequency in word_counts.items())
        scored.append((score, index, chunk))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected = [chunk for _, _, chunk in scored[:max_chunks]]
    if not selected:
        return chunks[:1]
    ordered = []
    seen = set()
    for _, index, chunk in sorted(scored[:max_chunks], key=lambda item: item[1]):
        marker = (index, chunk)
        if marker not in seen:
            ordered.append(chunk)
            seen.add(marker)
    return ordered if ordered else selected


def build_question_prompt(question: str, document_text: str) -> str:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    chunks = retrieve_relevant_chunks(document_text, cleaned_question)
    context_text = "\n\n---\n\n".join(chunks)
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS]

    return (
        "You are a document-grounded assistant. Answer using only the supplied document excerpts. "
        "If the answer is not supported by the document, say so clearly instead of guessing. "
        "Distinguish factual statements from reasonable interpretations.\n\n"
        f"Question: {cleaned_question}\n\n"
        f"Relevant document excerpts:\n{context_text}\n\n"
        "Answer concisely but accurately, and include source references such as chunk identifiers or page information when available."
    )


def build_summary_prompt(document_text: str) -> str:
    return (
        "Summarize the document in a structured format with the headings Overview, Main points, Key findings, "
        "Important dates, names, or figures, Risks or challenges, and Conclusion. Use only information contained in the document. "
        "Do not invent missing information.\n\nDocument:\n" + document_text
    )
