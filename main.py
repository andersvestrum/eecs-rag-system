#!/usr/bin/env python3
"""Autograder entrypoint: read questions, write predictions.

Usage:
    python3 main.py <questions_file> <predictions_file>
"""

import sys
import time

from rag import answer


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python3 main.py <questions_file> <predictions_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    questions_path = sys.argv[1]
    predictions_path = sys.argv[2]

    with open(questions_path) as f:
        questions = [line.strip() for line in f if line.strip()]

    start = time.time()
    predictions = []
    for i, q in enumerate(questions):
        q_start = time.time()
        pred = answer(q)
        elapsed = time.time() - q_start
        predictions.append(pred)
        print(
            f"[{i + 1}/{len(questions)}] ({elapsed:.1f}s) {q} -> {pred}",
            file=sys.stderr,
        )

    total = time.time() - start
    avg = total / max(len(questions), 1)
    print(
        f"Done: {len(predictions)} answers in {total:.1f}s ({avg:.1f}s/q)",
        file=sys.stderr,
    )

    with open(predictions_path, "w") as f:
        for pred in predictions:
            f.write(pred.replace("\n", " ").strip() + "\n")


if __name__ == "__main__":
    main()
