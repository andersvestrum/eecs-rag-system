#!/usr/bin/env python3
"""Autograder entrypoint: read questions, write predictions.

Usage:
    python3 main.py <questions_file> <predictions_file>

Output has exactly one line per input line, same order (required by autograder).
Per-question timeout returns "unknown" so a single timeout does not fail the run.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from rag import answer

# Per-question timeout (seconds). OpenRouter can occasionally hang.
QUESTION_TIMEOUT = 30


def _answer_with_timeout(question: str) -> str:
    """Run answer(question) in a thread; return 'unknown' on timeout."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(answer, question)
        try:
            return future.result(timeout=QUESTION_TIMEOUT)
        except FuturesTimeoutError:
            return "unknown"


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
        lines = f.readlines()

    start = time.time()
    predictions = []
    for i, line in enumerate(lines):
        q = line.rstrip("\n")
        if not q.strip():
            pred = "unknown"
        else:
            q_start = time.time()
            pred = _answer_with_timeout(q.strip())
            elapsed = time.time() - q_start
            q_display = (q[:57] + "...") if len(q) > 60 else q
            print(
                f"[{i + 1}/{len(lines)}] ({elapsed:.1f}s) {q_display} -> {pred}",
                file=sys.stderr,
            )
        # One answer per line; no newlines inside the answer (autograder requirement)
        predictions.append(pred.replace("\n", " ").strip() or "unknown")

    total = time.time() - start
    avg = total / max(len(lines), 1)
    print(
        f"Done: {len(predictions)} answers in {total:.1f}s ({avg:.1f}s/q)",
        file=sys.stderr,
    )

    with open(predictions_path, "w") as f:
        for pred in predictions:
            f.write(pred + "\n")


if __name__ == "__main__":
    main()
