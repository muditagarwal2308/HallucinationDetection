"""Prompt templates shared by all LLM back-ends."""

from __future__ import annotations

import re

QA_SYSTEM = "You are a concise, factual question-answering assistant."

GITA_SYSTEM = (
    "You answer questions about the Bhagavad Gita using only the verses provided. "
    "Cite every verse you rely on in the form (BG chapter.verse). If the verses do not answer the "
    "question, or the question rests on a false premise, say so instead of guessing."
)


def closed_book_prompt(question: str) -> str:
    return (
        "Answer the following question with a short answer (a name, number, date or a few words). "
        "Do not explain.\n\n"
        f"Question: {question}\nAnswer:"
    )


def rag_prompt(question: str, passages: list[str]) -> str:
    ctx = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return (
        "Use the passages below to answer the question with a short answer (a name, number, date or a few "
        "words). Do not explain.\n\n"
        f"Passages:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )


def p_true_prompt(question: str, answer: str, samples: list[str] | None = None) -> str:
    """Kadavath et al. (2022) self-evaluation prompt, optionally showing brainstormed samples."""
    brainstorm = ""
    if samples:
        brainstorm = "Here are some brainstormed ideas:\n" + "\n".join(samples[:10]) + "\n"
    return (
        f"Question: {question}\n{brainstorm}Possible answer: {answer}\n"
        "Is the possible answer:\n (A) True\n (B) False\n"
        "Reply with a single letter. The possible answer is: ("
    )


def gita_prompt(question: str, verses: list[tuple[str, str]]) -> str:
    ctx = "\n".join(f"(BG {ref}) {text}" for ref, text in verses)
    return (
        f"Verses:\n{ctx}\n\nQuestion: {question}\n"
        "Answer in 2-4 sentences, citing verses as (BG chapter.verse)."
    )


_PREFIX = re.compile(r"^(answer|final answer|a)\s*[:\-]\s*", re.IGNORECASE)


def clean_answer(text: str) -> str:
    """Keep the first line of a completion and strip boilerplate like ``Answer:``."""
    text = (text or "").strip()
    text = text.split("\n")[0].strip()
    text = _PREFIX.sub("", text).strip()
    text = text.strip("\"'` ")
    return text
