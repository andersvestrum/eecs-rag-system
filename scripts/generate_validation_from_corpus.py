#!/usr/bin/env python3
"""Generate 100+ validation Q&A pairs from the retrieval corpus.

Each answer is a short span that actually appears in a chunk, so the RAG
system can retrieve it and evaluation (EM/F1) is meaningful. Meets the
assignment requirement for "at least 100 questions" in your test data.

Run from repo root (after build_index):
  python -m scripts.generate_validation_from_corpus [--min-pairs 100] [--out-dir data]

Writes data/validation_questions.txt and data/validation_answers.txt (or --out-dir).
"""

import argparse
import json
import os
import re
from collections import defaultdict

# Paths relative to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHUNKS = os.path.join(REPO_ROOT, "data", "chunks.jsonl")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data")

# Patterns for short answer spans (assignment: under 10 words, extractive)
COURSE_RE = re.compile(
    r"\b(CS\s*\d+[A-Z]?|EE\s*\d+[A-Z]?|EECS\s*\d+[A-Z]?)\b",
    re.IGNORECASE
)
# Faculty name from title "Full Name | EECS at ..." or "Name, Title | ..."
TITLE_FACULTY_RE = re.compile(r"^([^|]+?)\s*\|\s*EECS", re.IGNORECASE)
# Date patterns (short answers)
DATE_SLASH_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b")
DATE_YEAR_RE = re.compile(r"\b(20\d{2}(?:-20\d{2})?)\b")
# Building names (from assignment / common EECS)
BUILDINGS = {"Soda Hall", "Cory Hall", "Jacobs Hall", "Sutardja Dai Hall"}
# Awards / phrases we can turn into "What award did X get?" (capture from chunk)
AWARD_LIKE_RE = re.compile(
    r"\b((?:Richard E\.\s*)?Merwin\s+[^.,]+|(?:Taylor L\.\s*)?Booth\s+[^.,]+|Grace Murray Hopper[^.,]*|Fellow\s+[^.,]+)",
    re.IGNORECASE
)


def _normalize_answer(a: str) -> str:
    a = a.strip()
    a = re.sub(r"\s+", " ", a)
    return a[:80]  # cap length


def _dedupe_answers(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep first occurrence per normalized answer to avoid duplicate questions."""
    seen = set()
    out = []
    for q, a in pairs:
        key = _normalize_answer(a).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((q, a))
    return out


def extract_course_pairs(chunk: dict) -> list[tuple[str, str]]:
    text = chunk.get("text") or ""
    title = chunk.get("title") or ""
    url = chunk.get("url") or ""
    pairs = []

    if "Course:" not in title and "/Courses/" not in url:
        return pairs

    code_from_title = re.search(r"Course:\s*((?:CS|EE|EECS)\s*\d+[A-Z]?)", title, re.I)
    codes_in_text = COURSE_RE.findall(text)
    code = None
    if code_from_title:
        code = code_from_title.group(1).strip()
    elif codes_in_text:
        code = codes_in_text[0].strip() if codes_in_text else None

    if not code:
        return pairs

    # Only emit if we have a real course description (not schedule/room info)
    schedule_room_re = re.compile(
        r"TuTh|MoWe|MoWeFr|WeFr|\d{1,2}:\d{2}|Cory\s*\d{3}|Evans\s*\d+|Tan\s*\d+|Valley Life Sciences|\d+\s*–\s*We\s|\d+\s*–\s*Mo"
    )
    desc = ""
    for m in re.finditer(r"[.\n]([^.\n]{20,120})", text):
        snippet = m.group(1).strip()
        if code in snippet and len(snippet) < 120:
            raw = snippet.replace(code, "").strip().strip(".-").strip()
            if not raw or raw.startswith("Course:") or len(raw) <= 10 or code in raw:
                continue
            if schedule_room_re.search(raw):
                continue  # skip schedule/room lines
            desc = raw[:60]
            break
    if not desc:
        return pairs

    q = f"What is the course number for the EECS course that {desc}?"
    pairs.append((q, code))
    return pairs


def extract_faculty_pairs(chunk: dict) -> list[tuple[str, str]]:
    text = (chunk.get("text") or "").strip()
    title = (chunk.get("title") or "").strip()
    url = (chunk.get("url") or "").lower()
    if not title or "|" not in title:
        return []
    if "/faculty" not in url and "homepage" not in url:
        return []
    m = TITLE_FACULTY_RE.match(title)
    if not m:
        return []
    name = m.group(1).strip()
    if re.match(r"^(Faculty|History|Honors|Courses|People|Course:|Our Staff|Technical Reports)\s", name, re.I):
        return []
    if re.match(r"^Course:", name, re.I) or "EECS at Berkeley" in name:
        return []
    if "," in name:
        name = name.split(",")[0].strip()
    if len(name) < 4 or len(name) > 50:
        return []
    if not re.search(r"[A-Za-z]{2,}\s+[A-Za-z]", name):
        return []

    # Question must NOT contain the answer (name). Use a clue from the chunk.
    award_m = AWARD_LIKE_RE.search(text)
    if award_m:
        clue = award_m.group(1).strip()[:50]
        q = f"Who is the EECS faculty member who received the {clue}?"
        return [(q, name)]
    # Try "research" / "focus" phrase that doesn't contain the name (skip noise)
    for m in re.finditer(r"(?:research|focus|work)[es]?\s+(?:on\s+)?([^.]{15,60})", text, re.I):
        phrase = m.group(1).strip()
        if name not in phrase and len(phrase) > 15:
            # Skip contact/support lines, emails, pure numbers
            if re.search(r"support\s|@|\.edu|^\d|Cory\s+\d|fax|phone", phrase, re.I):
                continue
            if re.match(r"^(Areas|Support|made\s+monumental)", phrase, re.I):
                continue
            q = f"Who is the EECS faculty member whose work involves {phrase[:50]}?"
            return [(q, name)]
    return []


def extract_date_pairs(chunk: dict) -> list[tuple[str, str]]:
    text = chunk.get("text") or ""
    pairs = []
    for m in DATE_SLASH_RE.finditer(text):
        date = m.group(1)
        start = max(0, m.start() - 80)
        snippet = text[start : m.end() + 40].lower()
        if "deadline" in snippet or "due" in snippet or "nomination" in snippet or "award" in snippet:
            q = "When is the deadline for the nomination of the outstanding TA awards?"
            pairs.append((q, date))
            break
    return pairs


def extract_building_pairs(chunk: dict) -> list[tuple[str, str]]:
    text = chunk.get("text") or ""
    pairs = []
    # Answer = location phrase with real place names (Hearst, LeRoy, Avenue, etc.)
    location_re = re.compile(
        r"intersection\s+of\s+([A-Za-z\s]+(?:and|&)\s+[A-Za-z\s]+(?:Avenue|Ave|Street|St|Road|Rd|Circle))",
        re.IGNORECASE
    )
    location_ok = re.compile(r"Avenue|Ave\.?|Street|St\.?|Road|Rd\.?|Hearst|LeRoy|Circle", re.I)
    for b in BUILDINGS:
        if b not in text:
            continue
        for m in location_re.finditer(text):
            loc = ("intersection of " + m.group(1).strip()).strip()
            if not location_ok.search(loc) or b in loc or "EECS" in loc or "Engineering" in loc:
                continue
            if 10 < len(loc) < 70:
                q = f"Where is {b} located on the UC Berkeley campus?"
                pairs.append((q, loc))
                return pairs
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate 100+ validation Q&A from corpus (answers in chunks)."
    )
    parser.add_argument(
        "--chunks",
        default=DEFAULT_CHUNKS,
        help="Path to chunks.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory for validation_questions.txt and validation_answers.txt",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=100,
        help="Minimum number of (question, answer) pairs to output",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.chunks):
        print(f"ERROR: Chunks file not found: {args.chunks}")
        print("Run scripts/build_index first (after crawl).")
        return 1

    all_pairs = []
    seen_questions = set()

    with open(args.chunks) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            for extractor in (
                extract_course_pairs,
                extract_faculty_pairs,
                extract_date_pairs,
                extract_building_pairs,
            ):
                for q, a in extractor(chunk):
                    qn = q.lower().strip()
                    if qn not in seen_questions and a and len(a) < 100:
                        seen_questions.add(qn)
                        all_pairs.append((q, a))

    # Dedupe by answer (keep one question per answer)
    all_pairs = _dedupe_answers(all_pairs)

    if len(all_pairs) < args.min_pairs:
        print(
            f"WARNING: Only {len(all_pairs)} unique pairs (requested {args.min_pairs}). "
            "Add more extractors or crawl more pages."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    qpath = os.path.join(args.out_dir, "validation_questions.txt")
    apath = os.path.join(args.out_dir, "validation_answers.txt")

    with open(qpath, "w") as fq, open(apath, "w") as fa:
        for q, a in all_pairs:
            # One per line; no newlines inside (assignment)
            fq.write(q.replace("\n", " ").strip() + "\n")
            fa.write(a.replace("\n", " ").strip() + "\n")

    print(f"Wrote {len(all_pairs)} validation pairs -> {qpath}")
    print(f"         (answers)              -> {apath}")
    return 0


if __name__ == "__main__":
    exit(main())
