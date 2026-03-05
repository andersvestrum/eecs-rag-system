#!/usr/bin/env python3
"""Download EECS web pages and save raw HTML.

Usage:
    python -m scripts.crawl
"""

import os
import json
import time
import hashlib
import requests
from bs4 import BeautifulSoup

OUT_DIR = os.path.join("data", "raw_html")
os.makedirs(OUT_DIR, exist_ok=True)

# Add seed URLs here (or load from a file)
SEED_URLS = [
    "https://eecs.berkeley.edu",
    # TODO: add more EECS pages
]


def slug(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def fetch(url: str, delay: float = 1.0) -> str | None:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        time.sleep(delay)
        return resp.text
    except Exception as e:
        print(f"  ✗ {url}: {e}")
        return None


def main():
    for url in SEED_URLS:
        print(f"Fetching {url}")
        html = fetch(url)
        if html is None:
            continue
        path = os.path.join(OUT_DIR, slug(url) + ".html")
        with open(path, "w") as f:
            f.write(html)
    print(f"Done — saved {len(os.listdir(OUT_DIR))} pages to {OUT_DIR}/")


if __name__ == "__main__":
    main()
