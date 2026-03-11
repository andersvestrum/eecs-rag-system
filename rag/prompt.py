"""Prompt construction and answer postprocessing for EECS factoid QA."""

import re

SYSTEM = (
    "You are a precise factoid question-answering system for UC Berkeley EECS. "
    "You will be given context passages and a question.\n\n"
    "Rules:\n"
    "- Read ALL context passages before answering.\n"
    "- Output ONLY the answer — no explanation, no sentence, no preamble.\n"
    "- Keep answers short: a name, number, date, or brief phrase (under 10 words).\n"
    "- Give exactly ONE answer.\n"
    "- Yes/no questions: answer 'Yes' or 'No'.\n"
    "- 'How many' questions: give a digit, e.g. '6'.\n"
    "- Only say 'unknown' as a last resort.\n\n"
    "Examples:\n"
    "Q: Where did Dan Klein get his PhD? A: Stanford University\n"
    "Q: What is the course number for the security class? A: CS 161\n"
    "Q: When is the TA award deadline? A: 2/18/26"
)

USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def build_prompt(question, chunks):
    """Return (system_prompt, user_query)."""
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    return SYSTEM, USER_TEMPLATE.format(context=context, question=question)


_WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

_PREFIX_STRIPS = (
    "Answer:", "A:", "The answer is:", "The answer is",
    "Based on the context,", "According to the context,",
)


def postprocess(raw):
    """Clean up raw LLM output into a short answer string."""
    ans = raw.strip().split("\n")[0].strip()
    ans = ans.replace("**", "").replace("*", "")

    for pfx in _PREFIX_STRIPS:
        if ans.lower().startswith(pfx.lower()):
            ans = ans[len(pfx):].strip()

    if len(ans) >= 2 and ans[0] == ans[-1] and ans[0] in ('"', "'"):
        ans = ans[1:-1].strip()

    if ", " in ans and len(ans.split(", ")) >= 3:
        ans = ans.split(", ")[0].strip()

    ans = re.sub(r"\s*\([^)]*\)\s*$", "", ans).strip()

    words = ans.split()
    if len(words) > 12:
        ans = " ".join(words[:10])

    if ans.lower() in _WORD_TO_NUM:
        ans = _WORD_TO_NUM[ans.lower()]

    if ans.endswith("."):
        ans = ans[:-1].strip()

    if not ans or ans.lower() in ("i don't know", "i don't know.", "n/a", "none"):
        return "unknown"

    return ans
