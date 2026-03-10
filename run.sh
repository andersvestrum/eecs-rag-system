#!/usr/bin/env bash
# Autograder entrypoint
# Usage: bash run.sh questions.txt predictions.txt
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python3 main.py "$1" "$2"
