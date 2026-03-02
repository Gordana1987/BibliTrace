"""
Scrape the full Bible (OT + NT) from Svetosavlje.org.
Single source: https://svetosavlje.org/sveto-pismo/
Run from backend: python scripts/scrape_bible.py
"""
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

INDEX_URL = "https://svetosavlje.org/sveto-pismo/"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "bible" / "bible.csv"
DELAY_SECONDS = 1.5

# Links to book pages: sveto-pismo-N/M/
BOOK_LINK_RE = re.compile(r"^https?://(?:www\.)?svetosavlje\.org/sveto-pismo-\d+/\d+/?$")
# Cross-refs like [*1 Мој 3, 22] — remove so we don't split on "1. " inside them
BRACKET_REF_RE = re.compile(r"\[[^\]]*\]")
# Verse start: after newline/start OR after space (verses often "1. text 2. text" on one line)
VERSE_START_RE = re.compile(r"(?:^|\n|\s)(\d+)\.\s+")

SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "sr,en;q=0.9",
    "Referer": "https://svetosavlje.org/",
}


def fetch(url: str, session: requests.Session) -> str:
    r = session.get(url, timeout=30, headers=SESSION_HEADERS)
    r.encoding = r.encoding or "utf-8"
    r.raise_for_status()
    return r.text


def extract_book_links(html: str) -> list[tuple[str, str]]:
    """Parse index page; return list of (url, book_name)."""
    soup = BeautifulSoup(html, "html.parser")
    entry = soup.find("div", class_="entry-content") or soup
    links = []
    seen = set()
    for a in entry.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue
        full = urljoin(INDEX_URL, href)
        parsed = urlparse(full)
        path = parsed.path.rstrip("/")
        if not re.match(r"^/sveto-pismo-\d+/\d+/?$", path):
            continue
        if full in seen:
            continue
        seen.add(full)
        name = a.get_text(strip=True)
        # Remove trailing "50 глава" etc.
        name = re.sub(r",\s*\d+\s*глав[ае]\s*$", "", name, flags=re.I).strip()
        name = re.sub(r"\s+", " ", name)
        if name:
            links.append((full, name))
    return links


def extract_main_text(soup: BeautifulSoup) -> str:
    """Get verse content from main body (entry-content)."""
    content = soup.find("div", class_="entry-content") or soup.find("section", class_="entry-body")
    if not content:
        return ""
    # Drop non-content blocks
    for tag in content.find_all(["script", "style", "nav"]):
        tag.decompose()
    for tag in content.find_all(class_=re.compile(r"pagination|share|entry-date|entry-meta|entry-footer")):
        tag.decompose()
    return content.get_text(separator="\n", strip=True)


def parse_verses_from_text(text: str, book_name: str) -> list[dict]:
    """
    Remove cross-refs [...], then find verse starts (N. at line start or after space).
    Chapter: when we see verse 1 after having seen verse 2+ in current chapter, increment chapter.
    """
    # Strip bracket refs so "[*1 Мој 22, 18.]" doesn't create false verse splits
    text_clean = BRACKET_REF_RE.sub(" ", text)
    rows = []
    matches = list(VERSE_START_RE.finditer(text_clean))
    current_chapter = 1
    last_verse_in_chapter = 0
    for i, mo in enumerate(matches):
        num = int(mo.group(1))
        if num < 1 or num > 250:
            continue
        start = mo.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text_clean)
        verse_text = text_clean[start:end].strip()
        if not verse_text or len(verse_text) < 2:
            continue
        # New chapter: verse 1 after we had higher verses in this chapter
        if num == 1 and last_verse_in_chapter >= 2:
            current_chapter += 1
        last_verse_in_chapter = num
        rows.append({
            "book": book_name,
            "chapter": current_chapter,
            "verse": num,
            "text": verse_text,
        })
    return rows


def parse_book_page(html: str, book_name: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    text = extract_main_text(soup)
    return parse_verses_from_text(text, book_name)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    print("Fetching index...")
    index_html = fetch(INDEX_URL, session)
    books = extract_book_links(index_html)
    print(f"Found {len(books)} book links")
    all_rows = []
    for i, (url, book_name) in enumerate(books):
        print(f"[{i + 1}/{len(books)}] {book_name[:50]}...")
        try:
            html = fetch(url, session)
            rows = parse_book_page(html, book_name)
            all_rows.extend(rows)
            print(f"  -> {len(rows)} verses")
        except Exception as e:
            print(f"  -> ERROR: {e}")
        if i < len(books) - 1:
            time.sleep(DELAY_SECONDS)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} verses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
