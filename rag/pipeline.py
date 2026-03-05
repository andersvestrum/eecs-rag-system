"""End-to-end RAG pipeline: question in → answer out."""

from .retrieve import retrieve
from .prompt import build_prompt, postprocess


def answer(question: str) -> str:
    """Return a short factoid answer for *question*."""
    chunks = retrieve(question)
    prompt = build_prompt(question, chunks)

    # Import llm at call-time so the module can be loaded without the
    # autograder's llm.py on the path during local development.
    from llm import llm
    raw = llm(prompt)

    return postprocess(raw)
