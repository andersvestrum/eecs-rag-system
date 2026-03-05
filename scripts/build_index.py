#!/usr/bin/env python3
"""Chunk raw HTML → embed → build FAISS index.

Usage:
    python -m scripts.build_index
"""

import json
import os

import faiss
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ── configuration ─────────────────────────────────────────────────────
EMBED_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE   = 512          # characters per chunk
CHUNK_OVERLAP = 64          # character overlap between chunks
RAW_DIR      = os.path.join("data", "raw_html")
DATA_DIR     = "data"

CHUNKS_PATH  = os.path.join(DATA_DIR, "chunks.jsonl")
INDEX_PATH   = os.path.join(DATA_DIR, "faiss.index")
META_PATH    = os.path.join(DATA_DIR, "meta.json")


# ── helpers ───────────────────────────────────────────────────────────
def extract_text(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def chunk_text(text: str, url: str) -> list[dict]:
    """Split *text* into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append({"url": url, "text": text[start:end]})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── main ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Chunk every HTML file
    all_chunks: list[dict] = []
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".html"):
            continue
        path = os.path.join(RAW_DIR, fname)
        with open(path) as f:
            html = f.read()
        text = extract_text(html)
        # Use filename as placeholder URL (you can map back later)
        url = fname.replace(".html", "")
        all_chunks.extend(chunk_text(text, url))

    print(f"Created {len(all_chunks)} chunks from {len(os.listdir(RAW_DIR))} files")

    # 2. Save chunks.jsonl
    with open(CHUNKS_PATH, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    # 3. Embed
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # 4. Build FAISS index (inner-product on normalised vecs ≈ cosine)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved FAISS index ({index.ntotal} vectors, dim={dim}) → {INDEX_PATH}")

    # 5. Save meta.json
    meta = {
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_chunks": len(all_chunks),
        "dim": dim,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta → {META_PATH}")


if __name__ == "__main__":
    main()
