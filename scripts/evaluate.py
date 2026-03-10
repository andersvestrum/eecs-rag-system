#!/usr/bin/env python3
"""Evaluate predictions using Exact Match and token-level F1.

Supports two reference formats:
  1. A .txt file with one answer per line (multiple valid answers separated by |)
  2. A .json file with a list of answer strings (| separated)

Usage:
    python -m scripts.evaluate <predictions.txt> <references.txt|json>
"""

import json
import re
import string
import sys
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower, remove articles/punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    np = normalize_answer(pred)
    return float(any(normalize_answer(g) == np for g in golds))


def token_f1(pred: str, golds: list[str]) -> float:
    pred_toks = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        gold_toks = normalize_answer(g).split()
        common = Counter(pred_toks) & Counter(gold_toks)
        n = sum(common.values())
        if n == 0:
            continue
        p = n / len(pred_toks) if pred_toks else 0
        r = n / len(gold_toks) if gold_toks else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        best = max(best, f1)
    return best


def load_references(path: str) -> list[str]:
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    with open(path) as f:
        return [line.strip() for line in f]


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.evaluate <predictions.txt> <references>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        preds = [line.strip() for line in f]

    refs = load_references(sys.argv[2])
    assert len(preds) == len(refs), (
        f"Count mismatch: {len(preds)} preds vs {len(refs)} refs"
    )

    total_em, total_f1 = 0.0, 0.0
    for i, (p, r) in enumerate(zip(preds, refs)):
        golds = [a.strip() for a in r.split("|")]
        em = exact_match(p, golds)
        f1 = token_f1(p, golds)
        total_em += em
        total_f1 += f1
        if em == 0:
            print(f"  [{i:>3}] MISS  pred={p!r:40s}  ref={r!r:40s}  F1={f1:.2f}")

    n = len(preds)
    print(f"\nResults ({n} questions):")
    print(f"  Exact Match : {total_em / n * 100:5.1f}%")
    print(f"  Token F1    : {total_f1 / n * 100:5.1f}%")


if __name__ == "__main__":
    main()
