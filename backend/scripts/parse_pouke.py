"""
Parse raw Поуке.орг HTML into bible.csv (no text editing beyond extraction).

Also writes verse_counts.json and parse_issues.json for integrity checks.

Run from backend/:
  python scripts/parse_pouke.py --corpus dk
  python scripts/parse_pouke.py --corpus dk --book 1
  python scripts/parse_pouke.py --corpus dk --compare-legacy data/stari_izvori/dk/bible.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pouke_common import (
    CANONICAL_BOOK_ORDER,
    CSV_COLUMNS,
    POUKE_BOOKS,
    filter_books,
    parse_chapter_html,
    raw_chapter_path,
)

BASE_DIR = Path(__file__).resolve().parents[1]


def parse_raw_corpus(
    raw_dir: Path,
    books: list,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = []

    for book in books:
        book_dir = raw_dir / f"book_{book.num:02d}"
        if not book_dir.is_dir():
            all_issues.append(
                {"book": book.canonical, "book_num": book.num, "issue": "missing_raw_dir"}
            )
            continue

        chapter_files = sorted(book_dir.glob("chapter_*.html"))
        if not chapter_files:
            all_issues.append(
                {"book": book.canonical, "book_num": book.num, "issue": "no_chapter_files"}
            )
            continue

        for path in chapter_files:
            chapter = int(path.stem.split("_")[1])
            html = path.read_text(encoding="utf-8")
            rows, issues = parse_chapter_html(html, book_name=book.canonical, chapter=chapter)
            all_rows.extend(rows)
            all_issues.extend(issues)

        n = sum(1 for r in all_rows if r["book"] == book.canonical)
        print(f"  #{book.num:02d} {book.canonical}: {len(chapter_files)} chapters → {n} verses")

    return all_rows, all_issues


def verse_counts_by_book(rows: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        c[r["book"]] += 1
    return dict(c)


def compare_legacy_counts(
    new_counts: dict[str, int],
    legacy_csv: Path,
    books: list,
) -> list[dict[str, Any]]:
    if not legacy_csv.exists():
        return [{"issue": "legacy_csv_missing", "path": str(legacy_csv)}]

    legacy = pd.read_csv(legacy_csv)
    diffs: list[dict[str, Any]] = []
    for book in books:
        new_n = new_counts.get(book.canonical, 0)
        legacy_n = int((legacy["book"] == book.legacy_dk).sum())
        if new_n != legacy_n:
            diffs.append(
                {
                    "book": book.canonical,
                    "legacy_label": book.legacy_dk,
                    "new_verses": new_n,
                    "legacy_verses": legacy_n,
                    "delta": new_n - legacy_n,
                }
            )
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse raw Поуке HTML → bible.csv")
    parser.add_argument("--corpus", default="dk", help="Corpus folder under data/ (default: dk)")
    parser.add_argument("--book", type=int, action="append", dest="books", help="Pouke book number")
    parser.add_argument("--from-book", type=int, help="Start at this book number (inclusive)")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: data/<corpus>/bible.csv)",
    )
    parser.add_argument(
        "--compare-legacy",
        type=Path,
        help="Compare verse totals with a legacy bible.csv (e.g. stari_izvori/dk)",
    )
    args = parser.parse_args()

    data_dir = BASE_DIR / "data" / args.corpus
    raw_dir = data_dir / "raw"
    output = args.output or (data_dir / "bible.csv")
    issues_path = data_dir / "parse_issues.json"
    counts_path = data_dir / "verse_counts.json"
    diff_path = data_dir / "verse_count_diff.json"

    books = filter_books(POUKE_BOOKS, only_nums=args.books, from_num=args.from_book)
    if not books:
        print("No books matched filters.")
        return

    print(f"Parsing {raw_dir} …")
    rows, issues = parse_raw_corpus(raw_dir, books)

    if not rows:
        print("No verses parsed.")
        if issues:
            issues_path.write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Issues → {issues_path}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df["_book_order"] = df["book"].map(CANONICAL_BOOK_ORDER)
    unknown = df[df["_book_order"].isna()]["book"].unique()
    if len(unknown):
        raise ValueError(f"Unknown book names (no canonical order): {list(unknown)}")
    df = df.sort_values(["_book_order", "chapter", "verse"], ignore_index=True)
    df = df.drop(columns=["_book_order"])
    df.to_csv(output, index=False, encoding="utf-8")

    counts = verse_counts_by_book(rows)
    counts_path.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    issues_path.write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(df)} verses → {output}")
    print(f"Verse counts → {counts_path}")
    print(f"Parse issues: {len(issues)} → {issues_path}")

    if issues:
        by_type = Counter(i.get("issue", i.get("warnings", "unknown")) for i in issues)
        print("  issue breakdown:", dict(by_type))

    legacy_path = args.compare_legacy or (BASE_DIR / "data" / "stari_izvori" / "dk" / "bible.csv")
    if args.corpus == "dk" and legacy_path:
        diffs = compare_legacy_counts(counts, legacy_path, books)
        diff_path.write_text(json.dumps(diffs, ensure_ascii=False, indent=2), encoding="utf-8")
        if diffs:
            print(f"⚠ Verse count mismatches vs legacy: {len(diffs)} → {diff_path}")
            for d in diffs[:10]:
                if "book" in d:
                    print(f"    {d['book']}: new={d['new_verses']} legacy={d['legacy_verses']} (Δ{d['delta']})")
        else:
            print(f"Verse counts match legacy ({legacy_path.name}) for all parsed books.")


if __name__ == "__main__":
    main()
