#!/usr/bin/env bash
# Build the CS 288 Assignment 3 submission zip.
# Run from repo root: bash scripts/build_submission.sh [output.zip]
# Requires: data/chunks.jsonl, data/faiss.index, data/bm25_corpus.json, data/meta.json
# (Generate these with: python -m scripts.crawl && python -m scripts.build_index)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

OUT="${1:-submission.zip}"
STAGING=".submission_staging"
rm -rf "$STAGING"
mkdir -p "$STAGING"

# Code and entrypoint
cp run.sh main.py llm.py requirements.txt "$STAGING/"
cp -r rag "$STAGING/"

# Optional: include scripts (for reference; autograder does not run them)
mkdir -p "$STAGING/scripts"
[ -f scripts/__init__.py ] && cp scripts/__init__.py "$STAGING/scripts/"
cp scripts/crawl.py scripts/build_index.py scripts/evaluate.py "$STAGING/scripts/" 2>/dev/null || true

# Data: retrieval datastore (required for RAG at test time)
mkdir -p "$STAGING/data"
for f in chunks.jsonl faiss.index bm25_corpus.json meta.json; do
  if [ -f "data/$f" ]; then
    cp "data/$f" "$STAGING/data/"
  else
    echo "WARNING: data/$f not found. Run crawl + build_index first." >&2
  fi
done

# QA data (required by assignment)
for f in example_questions.txt example_answers.txt; do
  [ -f "data/$f" ] && cp "data/$f" "$STAGING/data/"
done

# Ensure required data exists
for f in chunks.jsonl faiss.index bm25_corpus.json meta.json; do
  if [ ! -f "$STAGING/data/$f" ]; then
    echo "ERROR: $STAGING/data/$f missing. Generate with:" >&2
    echo "  python -m scripts.crawl --max-pages 2000 --delay 0.5" >&2
    echo "  python -m scripts.build_index --chunk-size 800 --overlap 200" >&2
    rm -rf "$STAGING"
    exit 1
  fi
done

# Create zip (flat structure so run.sh is at top level)
(cd "$STAGING" && zip -r "$REPO_ROOT/$OUT" . -x "*.pyc" -x "*__pycache__*")
rm -rf "$STAGING"

echo "Created $OUT"
echo "Contents:"
unzip -l "$OUT" | head -40
