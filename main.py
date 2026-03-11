#!/usr/bin/env python3
"""Usage: python3 main.py <questions_file> <predictions_file>"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from rag import answer

TIMEOUT = 30  # seconds per question


def _safe_answer(question):
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(answer, question)
        try:
            return fut.result(timeout=TIMEOUT)
        except FuturesTimeoutError:
            return "unknown"


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 main.py <questions> <predictions>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        lines = f.readlines()

    t0 = time.time()
    preds = []
    for i, line in enumerate(lines):
        q = line.strip()
        if not q:
            preds.append("unknown")
            continue
        t1 = time.time()
        pred = _safe_answer(q)
        dt = time.time() - t1
        display = (q[:57] + "...") if len(q) > 60 else q
        print(f"[{i+1}/{len(lines)}] ({dt:.1f}s) {display} -> {pred}", file=sys.stderr)
        preds.append(pred.replace("\n", " ").strip() or "unknown")

    elapsed = time.time() - t0
    print(f"Done: {len(preds)} answers in {elapsed:.1f}s ({elapsed/max(len(lines),1):.1f}s/q)", file=sys.stderr)

    with open(sys.argv[2], "w") as f:
        for p in preds:
            f.write(p + "\n")


if __name__ == "__main__":
    main()
