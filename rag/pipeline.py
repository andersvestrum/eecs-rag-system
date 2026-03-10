"""End-to-end RAG pipeline: question in -> short answer out."""

from .retrieve import retrieve
from .prompt import build_prompt, postprocess

FALLBACK = "unknown"


def answer(question: str) -> str:
    """Return a short factoid answer for *question*."""
    try:
        chunks = retrieve(question)
        prompt = build_prompt(question, chunks)

        from llm import llm
        raw = llm(prompt)

        return postprocess(raw)
    except Exception:
        return FALLBACK
