#!/usr/bin/env python3
"""Autograder entrypoint: read questions, write predictions.

Usage:
    python main.py <questions_file> <predictions_file>
"""

import sys
from rag import answer


def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <questions_file> <predictions_file>")
        sys.exit(1)

    questions_path = sys.argv[1]
    predictions_path = sys.argv[2]

    with open(questions_path) as f:
        questions = [line.strip() for line in f if line.strip()]

    predictions = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q}")
        pred = answer(q)
        predictions.append(pred)

    with open(predictions_path, "w") as f:
        for pred in predictions:
            f.write(pred + "\n")

    print(f"Wrote {len(predictions)} predictions → {predictions_path}")


if __name__ == "__main__":
    main()
