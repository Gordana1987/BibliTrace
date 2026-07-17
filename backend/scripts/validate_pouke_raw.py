"""
Scan raw Поуке HTML for download corruption (Cloudflare, truncated pages, etc.).

Run from backend/ before parse_pouke.py:
  python scripts/validate_pouke_raw.py --corpus dk
  python scripts/validate_pouke_raw.py --corpus dk --json data/dk/raw_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pouke_common import POUKE_BOOK_BY_NUM, validate_raw_chapter_html

BASE_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw Поуке chapter HTML")
    parser.add_argument("--corpus", default="dk")
    parser.add_argument("--json", type=Path, help="Write report JSON")
    args = parser.parse_args()

    raw_dir = BASE_DIR / "data" / args.corpus / "raw"
    if not raw_dir.is_dir():
        print(f"Missing {raw_dir}")
        return

    bad: list[dict] = []
    total = 0
    for path in sorted(raw_dir.rglob("chapter_*.html")):
        total += 1
        html = path.read_text(encoding="utf-8", errors="replace")
        issues = validate_raw_chapter_html(html)
        if issues:
            book_num = int(path.parent.name.split("_")[1])
            chapter = int(path.stem.split("_")[1])
            book = POUKE_BOOK_BY_NUM[book_num]
            bad.append(
                {
                    "path": str(path.relative_to(BASE_DIR / "data" / args.corpus)),
                    "book_num": book_num,
                    "book": book.canonical,
                    "chapter": chapter,
                    "issues": issues,
                }
            )

    print(f"Scanned {total} chapters, {len(bad)} failed validation.")
    for row in bad:
        print(f"  #{row['book_num']:02d} {row['book']} {row['chapter']}: {', '.join(row['issues'])}")

    out = args.json or (BASE_DIR / "data" / args.corpus / "raw_validation.json")
    out.write_text(json.dumps(bad, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report → {out}")

    if bad:
        print("\nRe-download bad chapters, e.g.:")
        books = sorted({row["book_num"] for row in bad})
        print(f"  python scripts/download_pouke.py --corpus {args.corpus} --force --book " + " --book ".join(str(b) for b in books))
        sys.exit(1)


if __name__ == "__main__":
    main()
