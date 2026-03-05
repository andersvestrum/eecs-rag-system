"""Prompt template and answer post-processing."""

SYSTEM = (
    "You are a helpful assistant that answers questions about the "
    "UC Berkeley EECS department. Use ONLY the provided context to answer. "
    "If the answer is not in the context, say 'I don't know'. "
    "Be concise — answer in one or two sentences."
)

TEMPLATE = """{system}

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Format retrieved chunks + question into a single LLM prompt."""
    context = "\n\n".join(
        f"[{c['url']}]\n{c['text']}" for c in chunks
    )
    return TEMPLATE.format(system=SYSTEM, context=context, question=question)


def postprocess(raw: str) -> str:
    """Clean up the raw LLM output into a final answer string."""
    answer = raw.strip()
    # Remove common prefixes the model might add
    for prefix in ("Answer:", "A:"):
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    return answer
