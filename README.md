# EECS RAG System — CS 288 Assignment 3

Retrieval-Augmented Generation system for answering factoid questions about UC Berkeley EECS.

## Architecture

```
question
  -> embed query (all-MiniLM-L6-v2)
  -> hybrid retrieval (FAISS dense + BM25 sparse, fused with RRF)
  -> top-k chunks injected into prompt
  -> LLM generates short answer via OpenRouter
  -> postprocess -> answer
```

## Setup

```bash
conda create -n rag python=3.10.12 -y && conda activate rag
pip install torch>=2.0.0 transformers>=4.30.0 sentence-transformers>=2.2.2 \
    faiss-cpu>=1.7.4 rank-bm25>=0.2.2 numpy>=1.24.0 tqdm>=4.64.0 \
    matplotlib>=3.5.0 pandas>=1.5.0 seaborn>=0.11.0

# For local scripts only (not needed by autograder):
pip install requests beautifulsoup4
```

## Offline Pipeline (run once locally)

```bash
# 1. Crawl EECS pages -> data/raw_html/ + data/url_map.json
python -m scripts.crawl --max-pages 2000 --delay 0.5

# 2. Chunk + embed + build indices -> data/{chunks.jsonl, faiss.index, bm25_corpus.json, meta.json}
python -m scripts.build_index --chunk-size 800 --overlap 200
```

## Run (autograder entrypoint)

```bash
export OPENROUTER_API_KEY="sk-..."  # for local testing
bash run.sh data/example_questions.txt data/predictions.txt
```

## Evaluate

```bash
python -m scripts.evaluate data/predictions.txt data/example_answers.txt
```

## Test run before submitting

From the repo root, with dependencies installed (e.g. `pip install -r requirements.txt` or `conda activate rag`):

```bash
# 1. Set your OpenRouter API key (required for real answers)
export OPENROUTER_API_KEY="sk-..."

# 2. Run the pipeline (same as autograder)
bash run.sh data/example_questions.txt data/predictions.txt

# 3. Evaluate against reference answers
python3 -m scripts.evaluate data/predictions.txt data/example_answers.txt
```

Without `OPENROUTER_API_KEY`, every answer will be `unknown` (the pipeline catches the error and falls back so it still finishes). Use the same Python that has `faiss`, `sentence_transformers`, etc. installed.

## Project Structure

```
├── run.sh                  # autograder entrypoint (calls main.py)
├── main.py                 # reads questions, writes predictions
├── llm.py                  # OpenRouter wrapper (autograder replaces this)
├── rag/
│   ├── __init__.py
│   ├── pipeline.py         # answer(question) -> string
│   ├── retrieve.py         # hybrid BM25 + FAISS retrieval
│   └── prompt.py           # prompt template + postprocessing
├── scripts/                # local-only (not used by autograder)
│   ├── crawl.py            # BFS crawler for eecs.berkeley.edu
│   ├── build_index.py      # chunk -> embed -> build indices
│   └── evaluate.py         # EM + F1 evaluation
└── data/
    ├── chunks.jsonl         # one chunk per line: {url, text, title}
    ├── faiss.index          # FAISS inner-product index
    ├── bm25_corpus.json     # tokenised corpus for BM25
    ├── meta.json            # embed model, chunk config
    ├── example_questions.txt
    └── example_answers.txt
```

[Offline]
  eecs.berkeley.edu → crawl.py → 1,492 HTML files
  HTML files → build_index.py → 15,270 chunks + FAISS index + BM25 corpus

[Runtime]
  question → retrieve.py → 5 best chunks (hybrid BM25 + FAISS + RRF)
  5 chunks + question → prompt.py → structured prompt
  prompt → llm.py → raw LLM output
  raw output → prompt.py/postprocess → clean short answer

## Creating the submission zip

1. Generate the retrieval datastore (if not already done):
   ```bash
   python -m scripts.crawl --max-pages 2000 --delay 0.5
   python -m scripts.build_index --chunk-size 800 --overlap 200
   ```
2. Build the zip (run from repo root):
   ```bash
   bash scripts/build_submission.sh submission.zip
   ```
3. Submit `submission.zip` to Gradescope. The zip contains: `run.sh`, code (`main.py`, `llm.py`, `rag/`), `requirements.txt`, and `data/` (chunks.jsonl, faiss.index, bm25_corpus.json, meta.json, plus any QA files).

## Submission Checklist

- [x] `run.sh` accepts `$1` (questions) and `$2` (predictions)
- [x] Uses `python3`, not `python`
- [x] `llm.py` is unmodified (autograder overwrites it)
- [x] No direct OpenRouter calls outside `llm.py`
- [x] All paths are relative
- [x] Output has same line count as input, one answer per line
- [x] Works within 4 GB RAM, no GPU
- [x] Timeout handling per question (falls back to "unknown")
- [ ] Ship `data/{chunks.jsonl, faiss.index, bm25_corpus.json, meta.json}` in zip (via `build_submission.sh`)

## Repo vs local

**In the GitHub repo:** code, `data/example_questions.txt`, `data/example_answers.txt`, `data/meta.json` (if present). The large generated files (`data/faiss.index`, `data/bm25_corpus.json`, `data/chunks.jsonl`) are not in the repo (they’re in `.gitignore`).

**To run locally:** generate those data files by running the offline pipeline once (crawl, then build_index) as in **Offline Pipeline** above.
