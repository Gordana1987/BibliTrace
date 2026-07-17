"""
Compare Поуке ijekav (dk) and ekavski (dk_ekav) bible.csv corpora.

Run from backend/ after both are parsed:
  python scripts/compare_pouke_dialects.py
  python scripts/compare_pouke_dialects.py --sample 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("book", "chapter", "verse", "text"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing column {col!r}")
    df["chapter"] = df["chapter"].astype(int)
    df["verse"] = df["verse"].astype(int)
    return df


def ref_key(row) -> tuple[str, int, int]:
    return (row["book"], int(row["chapter"]), int(row["verse"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Pouke ijekav vs ekavski CSV")
    parser.add_argument(
        "--ijekav",
        type=Path,
        default=BASE_DIR / "data" / "dk" / "bible.csv",
        help="Ijekav CSV (default: data/dk/bible.csv)",
    )
    parser.add_argument(
        "--ekav",
        type=Path,
        default=BASE_DIR / "data" / "dk_ekav" / "bible.csv",
        help="Ekav CSV (default: data/dk_ekav/bible.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data" / "dk_ekav" / "dialect_diff.json",
        help="Write detailed diff JSON",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=15,
        help="How many text-diff examples to print (0 = none)",
    )
    args = parser.parse_args()

    if not args.ijekav.exists():
        print(f"Missing ijekav CSV: {args.ijekav}")
        sys.exit(1)
    if not args.ekav.exists():
        print(f"Missing ekav CSV: {args.ekav}")
        print("Download and parse first:")
        print("  python scripts/download_pouke.py --corpus dk_ekav --resume --continue-on-error")
        print("  python scripts/parse_pouke.py --corpus dk_ekav")
        sys.exit(1)

    ijek = load_csv(args.ijekav)
    ekav = load_csv(args.ekav)

    ijek_map = {ref_key(r): r["text"] for _, r in ijek.iterrows()}
    ekav_map = {ref_key(r): r["text"] for _, r in ekav.iterrows()}

    ijek_refs = set(ijek_map)
    ekav_refs = set(ekav_map)
    common = ijek_refs & ekav_refs
    only_ijek = sorted(ijek_refs - ekav_refs)
    only_ekav = sorted(ekav_refs - ijek_refs)

    identical = 0
    different: list[dict] = []
    for ref in sorted(common):
        a, b = ijek_map[ref], ekav_map[ref]
        if a == b:
            identical += 1
        else:
            different.append(
                {
                    "book": ref[0],
                    "chapter": ref[1],
                    "verse": ref[2],
                    "ijekav": a,
                    "ekavski": b,
                }
            )

    book_counts_ijek = ijek.groupby("book").size().to_dict()
    book_counts_ekav = ekav.groupby("book").size().to_dict()
    count_diffs = []
    for book in sorted(set(book_counts_ijek) | set(book_counts_ekav)):
        ni = int(book_counts_ijek.get(book, 0))
        ne = int(book_counts_ekav.get(book, 0))
        if ni != ne:
            count_diffs.append({"book": book, "ijekav": ni, "ekavski": ne, "delta": ne - ni})

    report = {
        "ijekav_csv": str(args.ijekav),
        "ekav_csv": str(args.ekav),
        "ijekav_verses": len(ijek),
        "ekavski_verses": len(ekav),
        "common_refs": len(common),
        "only_ijekav": [{"book": r[0], "chapter": r[1], "verse": r[2]} for r in only_ijek],
        "only_ekavski": [{"book": r[0], "chapter": r[1], "verse": r[2]} for r in only_ekav],
        "identical_text": identical,
        "different_text": len(different),
        "book_verse_count_diffs": count_diffs,
        "text_diffs": different,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Поуке ijekav vs ekavski ===")
    print(f"Ijekav:   {len(ijek):>6} verses  ({args.ijekav})")
    print(f"Ekavski:  {len(ekav):>6} verses  ({args.ekav})")
    print(f"Common refs:     {len(common)}")
    print(f"Only ijekav:     {len(only_ijek)}")
    print(f"Only ekavski:    {len(only_ekav)}")
    print(f"Identical text:  {identical}")
    print(f"Different text:  {len(different)}")
    if count_diffs:
        print(f"Book count diffs: {len(count_diffs)}")
        for d in count_diffs:
            print(f"  {d['book']}: ijekav={d['ijekav']} ekavski={d['ekavski']} (Δ{d['delta']:+d})")
    print(f"\nFull report → {args.output}")

    if args.sample and different:
        print(f"\nSample text differences ({min(args.sample, len(different))}):")
        for row in different[: args.sample]:
            ref = f"{row['book']} {row['chapter']}:{row['verse']}"
            print(f"  {ref}")
            print(f"    ijekav:  {row['ijekav'][:100]}{'…' if len(row['ijekav']) > 100 else ''}")
            print(f"    ekavski: {row['ekavski'][:100]}{'…' if len(row['ekavski']) > 100 else ''}")


if __name__ == "__main__":
    main()
