"""Prompt template optimised for short factoid QA about UC Berkeley EECS."""

import re

SYSTEM = (
    "You are a precise factoid question-answering system for UC Berkeley EECS.\n"
    "You will be given context passages and a question.\n"
    "\n"
    "Instructions:\n"
    "1. Read ALL context passages carefully before answering.\n"
    "2. Find the specific entity, person, or fact the question asks about.\n"
    "3. Output ONLY the answer — no explanation, no full sentence, no preamble.\n"
    "4. Answers must be short: a single name, number, date, or brief phrase (under 10 words).\n"
    "5. Give exactly ONE answer — do NOT list multiple answers.\n"
    "6. For yes/no questions, answer exactly 'Yes' or 'No'.\n"
    "7. If the question asks 'how many', count carefully and give just the digit (e.g. '6' not 'six').\n"
    "8. If the question asks for a 'full name' or 'full title', give the complete name, not the abbreviation.\n"
    "9. Try hard to find the answer in the context. Only output 'unknown' as a last resort.\n"
    "\n"
    "Examples:\n"
    "Q: Where did Dan Klein get his PhD? A: Stanford University\n"
    "Q: What is the course number for the security class? A: CS 161\n"
    "Q: When is the TA award deadline? A: 2/18/26"
)

TEMPLATE = """{system}

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    return TEMPLATE.format(system=SYSTEM, context=context, question=question)


def postprocess(raw: str) -> str:
    """Clean LLM output into a concise answer string."""
    ans = raw.strip()

    # Take first line only
    ans = ans.split("\n")[0].strip()

    # Strip markdown formatting
    ans = ans.replace("**", "").replace("*", "")

    # Remove common prefixes
    for pfx in (
        "Answer:", "A:", "The answer is:", "The answer is",
        "Based on the context,", "According to the context,",
    ):
        if ans.lower().startswith(pfx.lower()):
            ans = ans[len(pfx):].strip()

    # Remove wrapping quotes
    if len(ans) >= 2 and ans[0] == ans[-1] and ans[0] in ('"', "'"):
        ans = ans[1:-1].strip()

    # If answer looks like a comma-separated list, take the first item
    if ", " in ans and len(ans.split(", ")) >= 3:
        ans = ans.split(", ")[0].strip()

    # Remove parenthetical clarifications like "(Belgium)" or "(2024)"
    ans = re.sub(r"\s*\([^)]*\)\s*$", "", ans).strip()

    # Truncate overly long answers (factoid answers should be under ~10 words)
    words = ans.split()
    if len(words) > 12:
        # Try to cut at a natural boundary
        for cutoff in [10, 8, 6]:
            truncated = " ".join(words[:cutoff])
            if truncated:
                ans = truncated
                break

    # Convert word numbers to digits
    word_to_num = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20",
    }
    if ans.lower() in word_to_num:
        ans = word_to_num[ans.lower()]

    # Remove trailing period
    if ans.endswith("."):
        ans = ans[:-1].strip()

    if not ans or ans.lower() in ("i don't know", "i don't know.", "n/a", "none"):
        ans = "unknown"

    return ans
