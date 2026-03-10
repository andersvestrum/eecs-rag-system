#!/usr/bin/env python3
"""Chunk crawled HTML, embed, and build FAISS + BM25 indices.

Reads HTML from data/raw_html/, uses data/url_map.json for URL mapping,
and writes:
  data/chunks.jsonl     – one JSON object per line {url, text, title}
  data/faiss.index      – FAISS inner-product index
  data/bm25_corpus.json – tokenised corpus for BM25
  data/meta.json        – configuration metadata

Usage:
    python -m scripts.build_index [--chunk-size 800] [--overlap 200]
"""

import argparse
import json
import os
import re

import faiss
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
RAW_DIR = os.path.join("data", "raw_html")
DATA_DIR = "data"
MAP_PATH = os.path.join(DATA_DIR, "url_map.json")

CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
BM25_PATH = os.path.join(DATA_DIR, "bm25_corpus.json")
META_PATH = os.path.join(DATA_DIR, "meta.json")

MIN_CHUNK_LEN = 40


# ── text extraction ──────────────────────────────────────────────────

def extract_text(html: str) -> str:
    """Extract readable text from HTML, preserving paragraph breaks."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "noscript", "iframe", "svg"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("title")
    return tag.get_text().strip() if tag else ""


# ── chunking ─────────────────────────────────────────────────────────

def chunk_text(
    text: str, url: str, title: str,
    chunk_size: int, overlap: int,
) -> list[dict]:
    """Split *text* into overlapping chunks, breaking at sentence
    boundaries when possible."""
    if len(text.strip()) < MIN_CHUNK_LEN:
        return []

    header = f"Page: {title}\n" if title else ""

    chunks: list[dict] = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # try to snap to a sentence boundary
        if end < len(text):
            for delim in ("\n\n", "\n", ". ", "? ", "! ", "; "):
                pos = text.rfind(delim, start + chunk_size // 2, end + 100)
                if pos != -1:
                    end = pos + len(delim)
                    break

        segment = text[start:end].strip()
        if len(segment) >= MIN_CHUNK_LEN:
            chunks.append({
                "url": url,
                "text": header + segment,
                "title": title,
            })

        next_start = end - overlap
        if next_start <= start:
            next_start = start + chunk_size
        start = next_start

    return chunks


# ── BM25 tokenisation ───────────────────────────────────────────────

def tokenize_bm25(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    url_map: dict[str, str] = {}
    if os.path.exists(MAP_PATH):
        with open(MAP_PATH) as f:
            url_map = json.load(f)

    if not os.path.isdir(RAW_DIR):
        print(f"No raw HTML found at {RAW_DIR}/ — run the crawler first.")
        return

    html_files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".html"))
    if not html_files:
        print("No .html files found — run the crawler first.")
        return

    # 1. Chunk all HTML files
    all_chunks: list[dict] = []
    for fname in html_files:
        file_id = fname.replace(".html", "")
        url = url_map.get(file_id, file_id)
        path = os.path.join(RAW_DIR, fname)
        with open(path, encoding="utf-8", errors="replace") as f:
            html = f.read()
        text = extract_text(html)
        title = extract_title(html)
        all_chunks.extend(
            chunk_text(text, url, title, args.chunk_size, args.overlap)
        )

    print(f"Created {len(all_chunks)} chunks from {len(html_files)} HTML files")
    if not all_chunks:
        print("Nothing to index.")
        return

    # 2. Save chunks.jsonl
    with open(CHUNKS_PATH, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved chunks -> {CHUNKS_PATH}")

    # 3. Save BM25 tokenised corpus
    bm25_corpus = [tokenize_bm25(c["text"]) for c in all_chunks]
    with open(BM25_PATH, "w") as f:
        json.dump(bm25_corpus, f)
    print(f"Saved BM25 corpus -> {BM25_PATH}")

    # 4. Embed with sentence-transformers
    print(f"Loading embedding model: {args.embed_model}")
    model = SentenceTransformer(args.embed_model)
    texts = [c["text"] for c in all_chunks]

    batch_size = 256
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_emb.append(emb)
        done = min(i + batch_size, len(texts))
        print(f"  Embedded {done}/{len(texts)}")

    embeddings = np.vstack(all_emb).astype("float32")

    # 5. Build FAISS index (inner-product on L2-normalised vecs ≈ cosine)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"Saved FAISS index ({index.ntotal} vecs, dim={dim}) -> {INDEX_PATH}")

    # 6. Save meta.json
    meta = {
        "embed_model": args.embed_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.overlap,
        "min_chunk_len": MIN_CHUNK_LEN,
        "num_chunks": len(all_chunks),
        "dim": dim,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta -> {META_PATH}")


if __name__ == "__main__":
    main()
