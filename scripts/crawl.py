#!/usr/bin/env python3
"""BFS crawler for eecs.berkeley.edu.

Crawls pages reachable from seed URLs, saves HTML, and records
a URL-to-file mapping.  Only follows links within the EECS domain
regex: https?://(?:www\\d*\\.)?eecs\\.berkeley\\.edu(?:/[^\\s]*)?

Usage:
    python -m scripts.crawl [--max-pages 2000] [--delay 0.5]
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

EECS_RE = re.compile(
    r"https?://(?:www\d*\.)?eecs\.berkeley\.edu(?:/[^\s]*)?"
)

SKIP_EXT = frozenset({
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".mp4", ".mp3", ".wav", ".zip", ".tar", ".gz", ".bz2", ".tgz",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".css", ".js", ".json", ".xml", ".rss", ".atom", ".ics",
    ".dmg", ".exe", ".msi", ".deb", ".rpm",
})

OUT_DIR = os.path.join("data", "raw_html")
MAP_PATH = os.path.join("data", "url_map.json")


def slug(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def is_eecs(url: str) -> bool:
    return bool(EECS_RE.fullmatch(url.split("#")[0].split("?")[0]))


def normalize(url: str) -> str:
    """Strip fragment, normalise trailing slash."""
    url = url.split("#")[0]
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    if parsed.query:
        return f"{parsed.scheme}://{parsed.netloc}{path}?{parsed.query}"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def should_skip(url: str) -> bool:
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext in SKIP_EXT:
        return True
    lower_path = parsed.path.lower()
    if any(kw in lower_path for kw in ("/login", "/signin", "/logout", "/cas/")):
        return True
    return False


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        full = urljoin(base_url, href)
        full = normalize(full)
        if is_eecs(full) and not should_skip(full):
            out.append(full)
    return out


def crawl(seeds: list[str], max_pages: int, delay: float):
    os.makedirs(OUT_DIR, exist_ok=True)

    visited: set[str] = set()
    queue: deque[str] = deque()
    url_map: dict[str, str] = {}

    for s in seeds:
        queue.append(normalize(s))

    sess = requests.Session()
    sess.headers["User-Agent"] = (
        "Mozilla/5.0 (compatible; CS288-RAG-Bot/1.0; "
        "+https://eecs.berkeley.edu)"
    )

    count = 0
    errors = 0

    while queue and count < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = sess.get(url, timeout=15, allow_redirects=True)
            resp.raise_for_status()

            ctype = resp.headers.get("Content-Type", "")
            if "text/html" not in ctype:
                continue

            html = resp.text
            count += 1

            fid = slug(url)
            with open(os.path.join(OUT_DIR, fid + ".html"), "w", encoding="utf-8") as f:
                f.write(html)
            url_map[fid] = url

            for link in extract_links(html, url):
                if link not in visited:
                    queue.append(link)

            if count % 50 == 0:
                print(f"  [{count:>5}] pages crawled  |  queue {len(queue)}")
                with open(MAP_PATH, "w") as mf:
                    json.dump(url_map, mf, indent=2)

            time.sleep(delay)

        except KeyboardInterrupt:
            print("\nInterrupted — saving progress.")
            break
        except Exception as e:
            errors += 1
            if errors < 20:
                print(f"  SKIP {url}: {e}")
            continue

    with open(MAP_PATH, "w") as f:
        json.dump(url_map, f, indent=2)

    print(f"\nFinished: {count} pages saved to {OUT_DIR}/")
    print(f"URL map ({len(url_map)} entries) -> {MAP_PATH}")
    print(f"Errors skipped: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Crawl eecs.berkeley.edu")
    parser.add_argument("--max-pages", type=int, default=2000)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    seeds = [
        "https://eecs.berkeley.edu",
        "https://eecs.berkeley.edu/academics/courses",
        "https://eecs.berkeley.edu/research",
        "https://eecs.berkeley.edu/people/faculty",
        "https://eecs.berkeley.edu/people/staff",
        "https://eecs.berkeley.edu/resources",
        "https://eecs.berkeley.edu/about",
        "https://www2.eecs.berkeley.edu/Courses/",
        "https://www2.eecs.berkeley.edu/Faculty/Homepages/",
        "https://www2.eecs.berkeley.edu/Pubs/Dissertations/",
        "https://www2.eecs.berkeley.edu/Scheduling/Commencement/",
        "https://eecs.berkeley.edu/resources/students/academic",
        "https://eecs.berkeley.edu/resources/students/prospective",
        "https://eecs.berkeley.edu/resources/grads",
    ]

    crawl(seeds, max_pages=args.max_pages, delay=args.delay)


if __name__ == "__main__":
    main()
