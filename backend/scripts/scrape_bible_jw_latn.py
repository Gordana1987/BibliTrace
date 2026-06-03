"""
Fetch Daničić–Karadžić Bible from JW.org (sr-latn) and save Cyrillic CSV.

Source (Latin, good ijekavian text):
  https://www.jw.org/sr-latn/biblioteka/sveto-pismo/dani%C4%8Di%C4%87-karad%C5%BEi%C4%87/knjige/

Each chapter is one HTML page with structured verses:
  <span class="verse" id="v1001001"> … </span>
Verse ids encode book/chapter/verse (e.g. v1001028 = Postanak 1:28).

We transliterate Latin → Cyrillic with latin_to_cyrillic.py (not JW's sr-cyrl).

Output: backend/data/bible/bible.csv  (book, chapter, verse, text)

Legal: JW/Watch Tower material — for personal/research use; check ToS before
redistributing the resulting CSV publicly.

Run from backend/:
  python scripts/scrape_bible_jw_latn.py --book postanak --max-chapters 1
  python scripts/scrape_bible_jw_latn.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Allow running as script from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from latin_to_cyrillic import latin_to_cyrillic

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "data" / "bible" / "bible.csv"
PARTIAL_PATH = BASE_DIR / "data" / "bible" / "bible_partial.csv"
PROGRESS_PATH = BASE_DIR / "data" / "bible" / "jw_scrape_progress.json"
CSV_COLUMNS = ["book", "chapter", "verse", "text"]

JW_ORIGIN = "https://www.jw.org"
EDITION_JSON = (
    "/sr-latn/biblioteka/sveto-pismo/dani%C4%8Di%C4%87-karad%C5%BEi%C4%87/knjige/json/"
)

SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BibliTrace/1.0; research corpus)",
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "sr,en;q=0.9",
}

VERSE_ID_RE = re.compile(r"^v(\d+)$")
LEADING_VERSE_NUM_RE = re.compile(r"^\d+\s*")


def parse_verse_id(verse_id: str) -> tuple[int, int, int]:
    """
    JW verse anchor ids: v1001001 (Gen 1:1), v66022021 (Rev 22:21).
    Format: book (1–2 digits) + chapter (3 digits) + verse (3 digits).
    """
    m = VERSE_ID_RE.match(verse_id)
    if not m:
        raise ValueError(f"Bad verse id: {verse_id!r}")
    digits = m.group(1)
    if len(digits) == 7:
        book = int(digits[0])
        chapter = int(digits[1:4])
        verse = int(digits[4:7])
    elif len(digits) == 8:
        book = int(digits[0:2])
        chapter = int(digits[2:5])
        verse = int(digits[5:8])
    else:
        raise ValueError(f"Unexpected verse id length: {verse_id!r}")
    return book, chapter, verse


def fetch_json(session: requests.Session, path: str) -> dict:
    url = JW_ORIGIN + path
    r = session.get(url, timeout=60, headers={**SESSION_HEADERS, "Accept": "application/json"})
    r.raise_for_status()
    return r.json()


def fetch_chapter_html(
    session: requests.Session,
    book_path: str,
    chapter: int,
    *,
    retries: int = 5,
    backoff: float = 5.0,
) -> str:
    url = JW_ORIGIN + book_path.rstrip("/") + f"/{chapter}/"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60, headers=SESSION_HEADERS)
            r.raise_for_status()
            r.encoding = r.encoding or "utf-8"
            return r.text
        except requests.RequestException as e:
            last_err = e
            if attempt < retries - 1:
                wait = backoff * (attempt + 1)
                print(f"    network error, retry {attempt + 1}/{retries} in {wait:.0f}s …")
                time.sleep(wait)
    assert last_err is not None
    raise last_err


def filter_books(
    books: list[dict],
    *,
    only_slugs: list[str] | None = None,
    after_slug: str | None = None,
    skip_slugs: set[str] | None = None,
) -> list[dict]:
    """Filter JW book list by slug include / resume point / already-done set."""
    if only_slugs:
        wanted = {s.lower() for s in only_slugs}
        books = [b for b in books if b["url_segment"].lower() in wanted]
    if skip_slugs:
        books = [b for b in books if b["url_segment"].lower() not in skip_slugs]
    if after_slug:
        slug = after_slug.lower()
        for i, book in enumerate(books):
            if book["url_segment"].lower() == slug:
                books = books[i:]
                break
        else:
            raise ValueError(f"Unknown book slug: {after_slug!r}")
    return books


def parse_chapter_verses(html: str) -> list[tuple[int, int, int, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[tuple[int, int, int, str]] = []
    for span in soup.find_all("span", class_="verse"):
        vid = span.get("id")
        if not vid:
            continue
        try:
            book, chapter, verse = parse_verse_id(vid)
        except ValueError:
            continue
        text = span.get_text(" ", strip=True)
        text = LEADING_VERSE_NUM_RE.sub("", text).strip()
        if not text:
            continue
        rows.append((book, chapter, verse, text))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows


def load_scrape_progress() -> set[str]:
    if not PROGRESS_PATH.exists():
        return set()
    data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    return {s.lower() for s in data.get("completed_books", [])}


def save_scrape_progress(completed: set[str]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps({"completed_books": sorted(completed)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_partial_rows() -> list[dict[str, Any]]:
    if not PARTIAL_PATH.exists():
        return []
    return pd.read_csv(PARTIAL_PATH).to_dict(orient="records")


def save_partial_rows(rows: list[dict[str, Any]]) -> None:
    PARTIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(PARTIAL_PATH, index=False, encoding="utf-8")


def load_books(session: requests.Session) -> list[dict]:
    data = fetch_json(session, EDITION_JSON)
    books_raw = data["editionData"]["books"]
    books: list[dict] = []
    for num_str, meta in sorted(books_raw.items(), key=lambda x: int(x[0])):
        books.append(
            {
                "book_num": int(num_str),
                "name_lat": meta["standardSingularBookName"],
                "url_segment": meta["urlSegment"],
                "url_path": meta["url"],
                "chapter_count": int(meta["chapterCount"]),
            }
        )
    return books


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape JW sr-latn DK Bible → Cyrillic CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output CSV (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--book",
        action="append",
        dest="books",
        metavar="SLUG",
        help="Only this urlSegment (e.g. postanak). Repeatable.",
    )
    parser.add_argument(
        "--max-chapters",
        type=int,
        default=0,
        help="Limit chapters per book (0 = all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between chapter requests (default: 2.0)",
    )
    parser.add_argument(
        "--after-book",
        metavar="SLUG",
        help="Start at this urlSegment (e.g. rimljanima), inclusive",
    )
    parser.add_argument(
        "--after-chapter",
        type=int,
        default=1,
        help="Chapter to start from for the first book when using --after-book",
    )
    parser.add_argument(
        "--latin-only",
        action="store_true",
        help="Keep Latin text (skip transliteration)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from bible_partial.csv + jw_scrape_progress.json",
    )
    args = parser.parse_args()

    session = requests.Session()
    all_books = load_books(session)
    completed = load_scrape_progress() if args.resume else set()
    try:
        all_books = filter_books(
            all_books,
            only_slugs=args.books,
            after_slug=args.after_book if not args.resume else None,
            skip_slugs=completed if args.resume else None,
        )
    except ValueError as e:
        print(e)
        return
    if not all_books:
        if args.resume:
            partial = load_partial_rows()
            if partial:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(partial, columns=CSV_COLUMNS).to_csv(
                    args.output, index=False, encoding="utf-8"
                )
                print(f"All books done — copied {len(partial)} verses → {args.output}")
                return
        print(f"No books matched filters: --book {args.books}")
        return

    all_rows: list[dict] = load_partial_rows() if args.resume else []
    if args.resume and completed:
        print(f"Resume: {len(completed)} books done, {len(all_rows)} verses in partial")
    total_chapters = sum(
        min(b["chapter_count"], args.max_chapters) if args.max_chapters else b["chapter_count"]
        for b in all_books
    )
    done = 0

    print(f"Books: {len(all_books)}, chapters to fetch: ~{total_chapters}")
    print(f"Output: {args.output}")

    for bi, book in enumerate(all_books, start=1):
        name_lat = book["name_lat"]
        name_out = name_lat if args.latin_only else latin_to_cyrillic(name_lat)
        ch_max = book["chapter_count"]
        if args.max_chapters:
            ch_max = min(ch_max, args.max_chapters)

        slug = book["url_segment"].lower()
        print(f"[{bi}/{len(all_books)}] {name_lat} ({slug}), {ch_max} chapters")

        start_ch = args.after_chapter if bi == 1 and args.after_book else 1
        book_rows: list[dict] = []
        for chapter in range(start_ch, ch_max + 1):
            done += 1
            html = fetch_chapter_html(session, book["url_path"], chapter)
            verses = parse_chapter_verses(html)
            for _book_num, ch, ver, text_lat in verses:
                text = text_lat if args.latin_only else latin_to_cyrillic(text_lat)
                row = {
                    "book": name_out,
                    "chapter": ch,
                    "verse": ver,
                    "text": text,
                }
                all_rows.append(row)
                book_rows.append(row)
            print(f"  ch {chapter}: {len(verses)} verses ({done}/{total_chapters})")
            if chapter < ch_max:
                time.sleep(args.delay)

        completed.add(slug)
        save_partial_rows(all_rows)
        save_scrape_progress(completed)
        print(f"  → book done: {len(book_rows)} verses (saved partial, total {len(all_rows)})")

    if not all_rows:
        print("No verses scraped.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved {len(df)} verses → {args.output}")
    if PARTIAL_PATH.exists() and args.output != PARTIAL_PATH:
        print(f"(partial backup remains at {PARTIAL_PATH})")


if __name__ == "__main__":
    main()
