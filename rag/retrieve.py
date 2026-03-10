"""Hybrid retrieval: BM25 (sparse) + FAISS (dense) with Reciprocal Rank Fusion.

Retrieves a large candidate set, then re-ranks by keyword relevance to keep
only the most useful chunks for the LLM prompt.
"""

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


def _tokenize(text: str) -> list[str]:
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
        corpus_tokens = [_tokenize(c["text"]) for c in _chunks]
        _bm25 = BM25Okapi(corpus_tokens)


def _rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion over multiple ranked lists."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def _keyword_overlap(query_tokens: list[str], chunk_text: str) -> float:
    """Score a chunk by how many query keywords appear in it."""
    chunk_lower = chunk_text.lower()
    chunk_words = set(_tokenize(chunk_lower))
    if not query_tokens:
        return 0.0
    hits = sum(1 for t in query_tokens if t in chunk_words)
    return hits / len(query_tokens)


def retrieve(query: str, top_k: int = 5, n_retrieve: int = 15) -> list[dict]:
    """Retrieve *n_retrieve* candidates, re-rank by keyword overlap,
    and return the best *top_k* for the LLM prompt."""
    _load()

    n_cand = min(n_retrieve * 10, len(_chunks))

    # Dense retrieval
    vec = _model.encode([query], normalize_embeddings=True)
    vec = np.asarray(vec, dtype="float32")
    _, faiss_ids = _faiss_index.search(vec, n_cand)
    dense_ranking = [int(i) for i in faiss_ids[0] if 0 <= i < len(_chunks)]

    # Sparse retrieval (BM25)
    tokens = _tokenize(query)
    bm25_scores = _bm25.get_scores(tokens)
    sparse_ranking = np.argsort(bm25_scores)[::-1][:n_cand].tolist()

    # Fuse with RRF
    fused = _rrf([dense_ranking, sparse_ranking])

    # First pass: collect candidates with URL dedup
    candidates: list[dict] = []
    url_counts: dict[str, int] = {}
    for idx in fused:
        if idx >= len(_chunks):
            continue
        chunk = _chunks[idx]
        url = chunk.get("url", "")
        cnt = url_counts.get(url, 0)
        if cnt >= 3:
            continue
        url_counts[url] = cnt + 1
        candidates.append(chunk)
        if len(candidates) >= n_retrieve:
            break

    # Second pass: re-rank candidates by keyword overlap with query
    query_tokens = _tokenize(query)
    scored = []
    for i, c in enumerate(candidates):
        kw_score = _keyword_overlap(query_tokens, c["text"])
        rrf_score = 1.0 / (1 + i)  # position score from RRF
        combined = 0.2 * kw_score + 0.8 * rrf_score
        scored.append((combined, i, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, _, c in scored[:top_k]]
