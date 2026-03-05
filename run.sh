#!/usr/bin/env bash
# Autograder entrypoint
# Usage: bash run.sh questions.txt predictions.txt
set -e
python main.py "$1" "$2"
