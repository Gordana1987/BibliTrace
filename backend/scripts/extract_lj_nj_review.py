"""
Find Latin words containing lj or nj (Gaj digraphs → љ / њ) before Cyrillic conversion.

Writes a review CSV with every occurrence so you can mark which tokens should stay
as digraphs and which should be split (l+j or n+j).

Run from backend/:
  # Full Bible (saves progress after each book; default delay 2s):
  python scripts/extract_lj_nj_review.py

  # Resume after a crash (loads partial CSV + skips finished books):
  python scripts/extract_lj_nj_review.py --resume

  # Start from a specific book (e.g. after manual failure at Rimljanima):
  python scripts/extract_lj_nj_review.py --after-book rimljanima --delay 2.0

  # From an existing Latin CSV:
  python scripts/extract_lj_nj_review.py --input data/dk/bible_latin.csv

Review columns (fill in by hand):
  manual_ok   yes = convert lj→љ / nj→њ as one letter (default)
              no  = split: lj→лj, nj→нj
  notes       free text

Output:
  data/dk/lj_nj_review.csv              — final, one row per occurrence
  data/dk/lj_nj_review_unique.csv       — final, one row per distinct word
  data/dk/lj_nj_review_partial.csv      — incremental while fetching
  data/dk/lj_nj_review_progress.json    — completed book slugs for --resume
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_bible_jw_latn import (  # noqa: E402
    fetch_chapter_html,
    filter_books,
    load_books,
    parse_chapter_verses,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = BASE_DIR / "data" / "dk" / "lj_nj_review.csv"
DEFAULT_UNIQUE_OUT = BASE_DIR / "data" / "dk" / "lj_nj_review_unique.csv"
PARTIAL_OUT = BASE_DIR / "data" / "dk" / "lj_nj_review_partial.csv"
PROGRESS_PATH = BASE_DIR / "data" / "dk" / "lj_nj_review_progress.json"

COLUMNS = ["book", "chapter", "verse", "word", "digraphs", "verse_text", "manual_ok", "notes"]

# Serbian Latin word token (keep apostrophe for elisions if any).
WORD_RE = re.compile(
    r"[A-Za-zčćđšžČĆĐŠŽ]+(?:'[A-Za-zčćđšžČĆĐŠŽ]+)?",
    re.UNICODE,
)


def digraphs_in_word(word: str) -> list[str]:
    """Return 'lj' and/or 'nj' if present (case-insensitive)."""
    lower = word.lower()
    found: list[str] = []
    if "lj" in lower:
        found.append("lj")
    if "nj" in lower:
        found.append("nj")
    return found


def scan_verse_row(book: str, chapter: int, verse: int, text: str) -> list[dict]:
    rows: list[dict] = []
    for match in WORD_RE.finditer(text):
        word = match.group(0)
        dgs = digraphs_in_word(word)
        if not dgs:
            continue
        rows.append(
            {
                "book": book,
                "chapter": chapter,
                "verse": verse,
                "word": word,
                "digraphs": ",".join(dgs),
                "verse_text": text,
                "manual_ok": "",
                "notes": "",
            }
        )
    return rows


def scan_dataframe(df: pd.DataFrame) -> list[dict]:
    required = {"book", "chapter", "verse", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    out: list[dict] = []
    for _, row in df.iterrows():
        out.extend(
            scan_verse_row(
                str(row["book"]),
                int(row["chapter"]),
                int(row["verse"]),
                str(row["text"]),
            )
        )
    return out


def load_progress() -> set[str]:
    if not PROGRESS_PATH.exists():
        return set()
    data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    return {s.lower() for s in data.get("completed_books", [])}


def save_progress(completed: set[str]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps({"completed_books": sorted(completed)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_partial_occurrences() -> list[dict]:
    if not PARTIAL_OUT.exists():
        return []
    df = pd.read_csv(PARTIAL_OUT)
    return df.to_dict(orient="records")


def save_partial(occurrences: list[dict]) -> None:
    PARTIAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(occurrences, columns=COLUMNS)
    df = df.sort_values(["book", "chapter", "verse", "word"], ignore_index=True)
    df.to_csv(PARTIAL_OUT, index=False, encoding="utf-8")


def fetch_and_scan_books(
    session: requests.Session,
    books: list[dict],
    *,
    max_chapters: int,
    delay: float,
    first_book_start_chapter: int,
    occurrences: list[dict],
    completed: set[str],
) -> list[dict]:
    """Fetch JW chapters, scan lj/nj words, save partial after each book."""
    for bi, book in enumerate(books, start=1):
        slug = book["url_segment"].lower()
        if slug in completed:
            print(f"Skipping {book['name_lat']} ({slug}) — already in progress file")
            continue

        name = book["name_lat"]
        ch_max = book["chapter_count"]
        if max_chapters:
            ch_max = min(ch_max, max_chapters)
        start_ch = first_book_start_chapter if bi == 1 else 1

        print(f"Fetching {name} ({slug}), chapters {start_ch}–{ch_max} …")
        book_rows: list[dict] = []

        for chapter in range(start_ch, ch_max + 1):
            html = fetch_chapter_html(session, book["url_path"], chapter)
            for _bn, ch, ver, text in parse_chapter_verses(html):
                book_rows.extend(scan_verse_row(name, ch, ver, text))
            if chapter < ch_max:
                time.sleep(delay)

        occurrences.extend(book_rows)
        completed.add(slug)
        save_partial(occurrences)
        save_progress(completed)
        print(f"  → {len(book_rows)} lj/nj hits; total {len(occurrences)} (saved partial)")

    return occurrences


def build_unique_summary(occurrences: list[dict]) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    first_loc: dict[str, str] = {}
    digraph_by_word: dict[str, str] = {}
    for row in occurrences:
        w = row["word"]
        counter[w] += 1
        if w not in first_loc:
            first_loc[w] = f"{row['book']} {row['chapter']}:{row['verse']}"
            digraph_by_word[w] = row["digraphs"]
    rows = [
        {
            "word": w,
            "digraphs": digraph_by_word[w],
            "occurrences": counter[w],
            "example_location": first_loc[w],
            "manual_ok": "",
            "notes": "",
        }
        for w in sorted(counter.keys(), key=str.lower)
    ]
    return pd.DataFrame(rows)


def write_final_outputs(occurrences: list[dict], output: Path, unique_output: Path) -> None:
    out_df = pd.DataFrame(occurrences, columns=COLUMNS)
    out_df = out_df.sort_values(["book", "chapter", "verse", "word"], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False, encoding="utf-8")
    build_unique_summary(occurrences).to_csv(unique_output, index=False, encoding="utf-8")
    print(f"Occurrences: {len(out_df)} → {output}")
    print(f"Unique words: {len(out_df['word'].unique())} → {unique_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract lj/nj Latin words for manual review before Cyrillic conversion"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Latin CSV (book, chapter, verse, text). If omitted, fetch from JW.",
    )
    parser.add_argument(
        "--book",
        action="append",
        dest="books",
        metavar="SLUG",
        help="JW urlSegment when fetching (e.g. postanak). Repeatable.",
    )
    parser.add_argument("--max-chapters", type=int, default=0)
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
        help="First chapter for the first book when using --after-book",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load partial CSV and skip books listed in progress JSON",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--unique-output", type=Path, default=DEFAULT_UNIQUE_OUT)
    args = parser.parse_args()

    if args.input:
        print(f"Reading {args.input} …")
        occurrences = scan_dataframe(pd.read_csv(args.input))
        if not occurrences:
            print("No lj/nj words found.")
            return
        write_final_outputs(occurrences, args.output, args.unique_output)
        print(
            "Fill manual_ok: yes = keep љ/њ digraph, no = split to l+j / n+j. "
            "Then tell me to apply your review."
        )
        return

    completed = load_progress() if args.resume else set()
    occurrences = load_partial_occurrences() if args.resume else []

    if args.resume and completed:
        print(f"Resume: {len(completed)} books done, {len(occurrences)} rows in partial")
    elif args.after_book:
        print(f"Starting from book slug: {args.after_book}")
        if args.after_chapter > 1:
            print(f"  first book from chapter {args.after_chapter}")

    session = requests.Session()
    all_books = load_books(session)
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
        if args.resume and occurrences:
            print("All books already fetched — writing final outputs from partial.")
            write_final_outputs(occurrences, args.output, args.unique_output)
        else:
            print("No books to fetch (check --book / --after-book / --resume).")
        return

    occurrences = fetch_and_scan_books(
        session,
        all_books,
        max_chapters=args.max_chapters,
        delay=args.delay,
        first_book_start_chapter=args.after_chapter if args.after_book else 1,
        occurrences=occurrences,
        completed=completed,
    )

    if not occurrences:
        print("No lj/nj words found.")
        return

    write_final_outputs(occurrences, args.output, args.unique_output)
    print(
        "Fill manual_ok: yes = keep љ/њ digraph, no = split to l+j / n+j. "
        "Then tell me to apply your review."
    )


if __name__ == "__main__":
    main()
