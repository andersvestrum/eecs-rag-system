"""Retrieve relevant chunks for a query (FAISS dense retrieval)."""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── paths (relative to repo root) ────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), os.pardir, "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
META_PATH  = os.path.join(DATA_DIR, "meta.json")

# ── lazy-loaded singletons ───────────────────────────────────────────
_index: faiss.Index | None = None
_chunks: list[dict] | None = None
_model: SentenceTransformer | None = None


def _load():
    """Load index, chunks, and embedding model once."""
    global _index, _chunks, _model

    if _index is not None:
        return

    with open(META_PATH) as f:
        meta = json.load(f)

    _model = SentenceTransformer(meta["embed_model"])
    _index = faiss.read_index(INDEX_PATH)

    _chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            _chunks.append(json.loads(line))


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Return the *top_k* most relevant chunks for *query*.

    Each returned dict has keys ``url`` and ``text``.
    """
    _load()

    vec = _model.encode([query], normalize_embeddings=True)
    vec = np.asarray(vec, dtype="float32")

    _, ids = _index.search(vec, top_k)
    return [_chunks[i] for i in ids[0] if i < len(_chunks)]
