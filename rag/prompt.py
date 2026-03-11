"""Prompt construction and answer postprocessing for EECS factoid QA."""

import re

SYSTEM = (
    "You answer factoid questions about UC Berkeley EECS using provided context.\n\n"
    "Rules:\n"
    "- Output ONLY the answer entity. No explanation, no sentence.\n"
    "- Be as short as possible: a name, number, date, or 1-5 word phrase.\n"
    "- 'How many' / 'how long' → just the number (e.g. '6', not '6 semesters').\n"
    "- Yes/no → answer exactly 'Yes' or 'No'.\n"
    "- 'Who' → give the full name.\n"
    "- 'What year' / 'which year' / 'when' → give just the year or date.\n"
    "- Never repeat the question. Never explain your reasoning.\n"
    "- Only say 'unknown' if the context truly has no relevant information.\n\n"
    "Q: Where did Dan Klein get his PhD? A: Stanford University\n"
    "Q: What is the course number for the security class? A: CS 161\n"
    "Q: When is the TA award deadline? A: 2/18/26\n"
    "Q: How many breadth areas must CS PhD students complete? A: 2\n"
    "Q: Who is the CS student advisor? A: Carol Marshall\n"
    "Q: Is the GRE required for admissions? A: No"
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

    ans = re.sub(r"\s*\([^)]*\)\s*$", "", ans).strip()

    # Trim verbose suffixes like "and by appointment", "and others"
    ans = re.sub(r'\s+and\s+(by\s+)?(?:appointment|others|more).*$', '', ans, flags=re.IGNORECASE).strip()

    words = ans.split()
    if len(words) > 15:
        ans = " ".join(words[:12])

    if ans.lower() in _WORD_TO_NUM:
        ans = _WORD_TO_NUM[ans.lower()]

    # Strip trailing unit words from numeric answers (e.g. "30 hours" -> "30")
    m = re.match(r'^(\d+[\d,./]*)\s+(hours?|minutes?|semesters?|years?|units?|credits?|months?|days?|weeks?)$', ans, re.IGNORECASE)
    if m:
        ans = m.group(1)

    if ans.endswith("."):
        ans = ans[:-1].strip()

    if not ans or ans.lower() in ("i don't know", "i don't know.", "n/a", "none", "unknown"):
        return "unknown"

    return ans
