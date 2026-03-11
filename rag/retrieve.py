"""Hybrid BM25 + FAISS retrieval with Reciprocal Rank Fusion."""

import json
import os
import re
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
BM25_PATH = os.path.join(DATA_DIR, "bm25_corpus.json")
META_PATH = os.path.join(DATA_DIR, "meta.json")

_faiss_index = None
_bm25 = None
_chunks = None
_model = None


def _tokenize(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _load():
    global _faiss_index, _bm25, _chunks, _model
    if _chunks is not None:
        return

    with open(META_PATH) as f:
        meta = json.load(f)

    _model = SentenceTransformer(meta["embed_model"])
    _faiss_index = faiss.read_index(INDEX_PATH)

    _chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            _chunks.append(json.loads(line))

    if os.path.exists(BM25_PATH):
        with open(BM25_PATH) as f:
            corpus_tokens = json.load(f)
        _bm25 = BM25Okapi(corpus_tokens)
    else:
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _chunks])


def _rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _keyword_overlap(qtoks, text):
    words = set(_tokenize(text))
    if not qtoks:
        return 0.0
    return sum(1 for t in qtoks if t in words) / len(qtoks)


def retrieve(query, top_k=5, n_retrieve=15):
    _load()
    n_cand = min(n_retrieve * 10, len(_chunks))

    vec = _model.encode([query], normalize_embeddings=True)
    vec = np.asarray(vec, dtype="float32")
    _, faiss_ids = _faiss_index.search(vec, n_cand)
    dense_ranking = [int(i) for i in faiss_ids[0] if 0 <= i < len(_chunks)]

    tokens = _tokenize(query)
    bm25_scores = _bm25.get_scores(tokens)
    sparse_ranking = np.argsort(bm25_scores)[::-1][:n_cand].tolist()

    fused = _rrf([dense_ranking, sparse_ranking])

    # Collect candidates, limiting to 3 chunks per source URL
    candidates = []
    url_counts = {}
    for idx in fused:
        if idx >= len(_chunks):
            continue
        chunk = _chunks[idx]
        url = chunk.get("url", "")
        if url_counts.get(url, 0) >= 3:
            continue
        url_counts[url] = url_counts.get(url, 0) + 1
        candidates.append(chunk)
        if len(candidates) >= n_retrieve:
            break

    # Re-rank by keyword overlap blended with RRF position
    qtoks = _tokenize(query)
    scored = [
        (0.2 * _keyword_overlap(qtoks, c["text"]) + 0.8 / (1 + i), i, c)
        for i, c in enumerate(candidates)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, _, c in scored[:top_k]]
