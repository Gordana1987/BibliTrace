"""
Scrape the Bakotić Serbian Bible (OT + NT, Ekavian) from Serbian Wikisource.

Step 1 version: fetch all books and save a plain verse CSV only
  backend/data/bakotic/bible.csv

This mirrors the Daničić–Karadžić CSV layout:
  book, chapter, verse, text

Later steps (lemmatization, BM25, embeddings, multi-version support)
will build on this CSV, but for now we only care about getting clean text.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://sr.wikisource.org"
TOC_URL = BASE_URL + "/sr-ec/Библија_(Бакотић)"

# Output location: backend/data/bakotic/bible.csv
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
BAKOTIC_DIR = DATA_DIR / "bakotic"
OUTPUT_PATH = BAKOTIC_DIR / "bible.csv"

DELAY_SECONDS = 1.5

# Chapter headings look like: "Глава 1." (sometimes without the dot).
CHAPTER_RE = re.compile(r"^\s*Глава\s+(\d+)\.?\s*$")

# Verse lines look like: "1 У почетку створи Бог..." (verse number + space, sometimes with a trailing dot or ")")
VERSE_RE = re.compile(r"^\s*(\d+)[\.\)]?\s+(.*\S)\s*$")

SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "sr,en;q=0.9",
    "Referer": "https://sr.wikisource.org/",
}


def fetch(url: str, session: requests.Session) -> str:
    r = session.get(url, timeout=30, headers=SESSION_HEADERS)
    r.encoding = r.encoding or "utf-8"
    r.raise_for_status()
    return r.text


def build_absolute_url(href: str) -> str:
    """Convert a Wikisource href to an absolute URL."""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if not href.startswith("/"):
        href = "/" + href
    return BASE_URL + href


def parse_book(html: str, book_name: str) -> list[dict]:
    """
    Parse one Bakotić book HTML into verse rows.

    Strategy:
      - Extract visible text from the main content area.
      - Split into lines.
      - Lines matching "Глава N." set current chapter.
      - Lines starting with "N " (or "N." / "N)") are verses within that chapter.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Use main content container if available, fall back to whole document.
    content = soup.find("div", id="mw-content-text") or soup
    text = content.get_text(separator="\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Some books (like Авдија) have no "Глава N." headings; others do.
    has_chapters = any(CHAPTER_RE.match(ln) for ln in lines)

    rows: list[dict] = []
    current_chapter: int | None = None if has_chapters else 1

    for line in lines:
        # First, look for explicit chapter headings (when present).
        mch = CHAPTER_RE.match(line)
        if mch:
            current_chapter = int(mch.group(1))
            continue

        m = VERSE_RE.match(line)
        if not m:
            continue
        if current_chapter is None:
            # We haven't seen a chapter heading yet in a book that uses them;
            # skip verse-like noise before "Глава 1."
            continue

        verse_num = int(m.group(1))
        verse_text = m.group(2).strip()
        if not verse_text:
            continue

        rows.append(
            {
                "book": book_name,
                "chapter": current_chapter,
                "verse": verse_num,
                "text": verse_text,
            }
        )
    return rows


def main() -> None:
    BAKOTIC_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    # Step 1: fetch TOC and discover all Bakotić book links dynamically,
    # instead of relying on hard-coded names. This avoids 404s when
    # link texts differ (e.g. "Павлова посланица Римљанима").
    print(f"Fetching TOC from {TOC_URL} ...")
    toc_html = fetch(TOC_URL, session)
    toc_soup = BeautifulSoup(toc_html, "html.parser")

    book_links: list[tuple[str, str]] = []
    seen = set()
    for a in toc_soup.find_all("a", href=True):
        href = a["href"]
        # We only want links into the Bakotić Bible namespace.
        # On Wikisource these hrefs are percent-encoded, e.g.
        # "/wiki/%D0%91%D0%B8%D0%B1%D0%BB%D0%B8%D1%98%D0%B0_(%D0%91%D0%B0%D0%BA%D0%BE%D1%82%D0%B8%D1%9B)_:_%D0%9F%D0%BE%D1%81%D1%82%D0%B0%D1%9A%D0%B5"
        if "/wiki/%D0%91%D0%B8%D0%B1%D0%BB%D0%B8%D1%98%D0%B0_(%D0%91%D0%B0%D0%BA%D0%BE%D1%82%D0%B8%D1%9B)_:_" not in href:
            continue
        full_url = build_absolute_url(href)
        title = a.get_text(strip=True)
        if not title or full_url in seen:
            continue
        seen.add(full_url)
        book_links.append((title, full_url))

    if not book_links:
        print("No Bakotić book links found on TOC page; aborting.")
        return

    print(f"Discovered {len(book_links)} Bakotić book pages.")

    all_rows: list[dict] = []
    print(f"Scraping Bakotić Bible to {OUTPUT_PATH} ...")

    for idx, (book, url) in enumerate(book_links, start=1):
        print(f"[{idx}/{len(book_links)}] {book} -> {url}")
        try:
            html = fetch(url, session)
            rows = parse_book(html, book)
            all_rows.extend(rows)
            print(f"  -> {len(rows)} verses")
        except Exception as e:
            print(f"  -> ERROR for {book}: {e}")
        # Be gentle with Wikisource; brief pause between books.
        if idx < len(book_links):
            time.sleep(DELAY_SECONDS)

    if not all_rows:
        print("No verses scraped; nothing to write.")
        return

    df = pd.DataFrame(all_rows, columns=["book", "chapter", "verse", "text"])
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(df)} verses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
