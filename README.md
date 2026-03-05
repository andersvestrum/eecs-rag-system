# EECS RAG System — CS 288 NLP

A Retrieval-Augmented Generation (RAG) system for answering questions about the UC Berkeley EECS department.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the autograder entrypoint
bash run.sh questions.txt predictions.txt
```

## Project Structure

```
├── run.sh                # autograder entrypoint (calls main.py)
├── main.py               # reads questions, writes predictions
├── llm.py                # PROVIDED by autograder (do not modify)
│
├── rag/
│   ├── __init__.py
│   ├── pipeline.py       # answer(question) -> string
│   ├── retrieve.py       # FAISS dense retrieval
│   └── prompt.py         # prompt template + postprocess
│
├── scripts/              # run locally (not used by autograder)
│   ├── crawl.py          # download EECS pages
│   └── build_index.py    # chunk -> embed -> FAISS index
│
└── data/                 # ship these with submission
    ├── chunks.jsonl       # one chunk per line: {"url","text"}
    ├── faiss.index        # FAISS retrieval index
    └── meta.json          # config: embed model, chunk size, etc.
```

## Offline Pipeline (run once locally)

```bash
# 1. Crawl EECS pages → data/raw_html/
python -m scripts.crawl

# 2. Chunk + embed + build index → data/{chunks.jsonl, faiss.index, meta.json}
python -m scripts.build_index
```

## Autograder Pipeline

```bash
bash run.sh questions.txt predictions.txt
```

This reads each question, retrieves relevant chunks via FAISS, builds a
prompt with context, calls `llm()`, and writes the answer.

## Runtime Flow

```
question → embed query → FAISS search → top-k chunks → build prompt → llm() → answer
```

Models are loaded once (lazily on first call). Index and chunks are pre-built.
