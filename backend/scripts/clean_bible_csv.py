"""
Post-process backend/data/dk/bible.csv to remove DK editorial headlines.

Chapter headings on svetosavlje.org are formatted as "N. Phrase1. Phrase2." —
period-separated noun phrases summarising the chapter. The scraper captures them
as fake verses (verse N of the previous chapter). This script removes them.

Run from backend root:
    python scripts/clean_bible_csv.py --extract-review
        Writes review CSVs only — does NOT modify bible.csv.

    python scripts/clean_bible_csv.py --dry-run
        Print counts and a sample (bible.csv unchanged).

    python scripts/clean_bible_csv.py --apply
        Heuristic-only rewrite (strip pagination + remove all detected headline rows).
        Does not read manual_ok — risky if the heuristic mis-fires on real verses.

    python scripts/clean_bible_csv.py --apply-reviewed [--review-csv PATH]
        Apply decisions from a reviewed candidates_all.csv:
        - Pagination strip: default yes if manual_ok is empty; skip if manual_ok is no.
        - Row removal: only if manual_ok is explicitly yes (yes/y/true/1).

Also removes leftover "Pages: N" pagination rows / suffixes.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "dk" / "bible.csv"
REVIEW_DIR = Path(__file__).resolve().parent.parent / "data" / "dk" / "review_cleanup"

# Detects ". CAPITAL-LETTER" inside verse text (internal sentence boundary).
_INTERNAL_SENTENCE_RE = re.compile(r"\.\s+[\u0400-\u04FF\u0410-\u044FA-Z]")

# If the segment after ". " starts with one of these words, it is a real verse continuation
# (coordinating conjunctions, demonstratives, common verbal forms) — NOT a heading.
_VERSE_CONTINUATION_RE = re.compile(
    r"\.\s+(?:"
    r"А|И|Али|Него|Јер|Кад|Када|Па|Потом|Тада|Зато|Онда|Ако|Те|Јаох|Гле|Ево|Затим"
    r"|То|Ово|Оно|Они|Оне|Тако|Нека|Ту|Ни|Није|Нису|Да\b|Или|Сад|Та\b|Тад\b"
    r"|Ти\b|Сви\b|Рече\b|Има\b|Не\b|Оба\b|Биће\b|Свако\b|Свак\b"
    r")\b"
)


def _is_chapter_heading(text: str) -> bool:
    """Return True if the text looks like a chapter heading rather than real verse content.

    Chapter headings are short, noun-phrase lists without commas/semicolons, and
    contain at least one internal sentence boundary ('. CAPITAL') where the new
    phrase starts with something other than a coordinating conjunction.
    """
    if len(text) > 150:
        return False
    if ";" in text or "," in text:
        return False
    if not _INTERNAL_SENTENCE_RE.search(text):
        return False
    if _VERSE_CONTINUATION_RE.search(text):
        return False
    return True


def _strip_pagination(text: str) -> str:
    """Strip embedded 'Pages:\\n1\\n2...' suffix from verse text."""
    idx = text.find("\nPages:")
    return text[:idx].strip() if idx != -1 else text


def _is_pagination_artifact(text: str) -> bool:
    """Return True for standalone pagination rows like 'Pages:\\n1\\n2\\n3'."""
    stripped = text.strip()
    return stripped.startswith("Pages:") or re.match(r"^[\d\s]+$", stripped) is not None


def _prepare_processed_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with embedded \\nPages: suffix stripped from text (same as full cleanup)."""
    out = df.copy()
    embedded = out["text"].str.contains("\nPages:", na=False)
    out.loc[embedded, "text"] = out.loc[embedded, "text"].apply(_strip_pagination)
    return out


def run_extract_review(review_dir: Path) -> None:
    """Write CSV files for manual review. Never modifies bible.csv."""
    df = pd.read_csv(CSV_PATH, dtype=str)
    review_dir.mkdir(parents=True, exist_ok=True)

    df_orig = df.copy()
    df_orig["_csv_row"] = range(2, len(df_orig) + 2)  # 1-based line in file; +1 for header row

    df_proc = _prepare_processed_frame(df_orig.drop(columns=["_csv_row"], errors="ignore"))
    df_proc["_csv_row"] = df_orig["_csv_row"].values

    embedded_mask = df_orig["text"].str.contains("\nPages:", na=False)
    heading_mask = df_proc["text"].apply(_is_chapter_heading)
    pagination_mask = df_proc["text"].apply(_is_pagination_artifact)

    rows: list[dict] = []

    for idx in df_proc.index:
        kinds: list[str] = []
        if embedded_mask.loc[idx]:
            kinds.append("strip_pagination_suffix")
        if heading_mask.loc[idx]:
            kinds.append("remove_row_headline_heuristic")
        if pagination_mask.loc[idx] and not heading_mask.loc[idx]:
            kinds.append("remove_row_standalone_pagination")

        if not kinds:
            continue

        text_orig = str(df_orig.loc[idx, "text"])
        text_proc = str(df_proc.loc[idx, "text"])
        primary = (
            "remove_row_headline_heuristic"
            if heading_mask.loc[idx]
            else (
                "remove_row_standalone_pagination"
                if pagination_mask.loc[idx]
                else "strip_pagination_suffix"
            )
        )
        rows.append(
            {
                "csv_data_line": int(df_proc.loc[idx, "_csv_row"]),
                "primary_action": primary,
                "all_flags": ";".join(kinds),
                "book": df_proc.loc[idx, "book"],
                "chapter": df_proc.loc[idx, "chapter"],
                "verse": df_proc.loc[idx, "verse"],
                "text_original": text_orig,
                "text_after_pagination_strip": text_proc,
                "remove_row": "yes" if (heading_mask.loc[idx] or pagination_mask.loc[idx]) else "no",
                "manual_ok": "",
                "manual_notes": "",
            }
        )

    review_all = pd.DataFrame(rows)
    review_all.sort_values(["book", "chapter", "verse"], inplace=True, kind="stable")

    out_all = review_dir / "candidates_all.csv"
    review_all.to_csv(out_all, index=False, encoding="utf-8")
    print(f"Wrote {len(review_all)} candidate rows to {out_all}")
    print("  Columns manual_ok / manual_notes: fill after review (e.g. yes/no).")

    only_headlines = review_all[review_all["primary_action"] == "remove_row_headline_heuristic"].copy()
    out_h = review_dir / "candidates_headlines_only.csv"
    only_headlines.to_csv(out_h, index=False, encoding="utf-8")
    print(f"Wrote {len(only_headlines)} headline-removal candidates to {out_h}")

    only_strip = review_all[review_all["primary_action"] == "strip_pagination_suffix"].copy()
    out_s = review_dir / "candidates_pagination_strip_only.csv"
    only_strip.to_csv(out_s, index=False, encoding="utf-8")
    print(f"Wrote {len(only_strip)} pagination-suffix-only rows to {out_s}")

    only_standalone = review_all[
        review_all["primary_action"] == "remove_row_standalone_pagination"
    ].copy()
    out_p = review_dir / "candidates_standalone_pagination_rows.csv"
    only_standalone.to_csv(out_p, index=False, encoding="utf-8")
    print(f"Wrote {len(only_standalone)} standalone pagination row candidates to {out_p}")

    print(
        "\nbible.csv was not modified. After review, run:\n"
        "  python scripts/clean_bible_csv.py --apply-reviewed\n"
        "or (heuristic only, no manual_ok): python scripts/clean_bible_csv.py --apply"
    )


def _cell_str(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _manual_explicit_yes(val: object) -> bool:
    return _cell_str(val).lower() in ("yes", "y", "true", "1")


def _manual_explicit_no(val: object) -> bool:
    return _cell_str(val).lower() in ("no", "n", "false", "0")


def run_apply_reviewed(review_csv: Path) -> None:
    """Rewrite bible.csv using reviewed candidates (conservative row removal)."""
    if not review_csv.is_file():
        raise SystemExit(f"Review CSV not found: {review_csv}")

    df = pd.read_csv(CSV_PATH, dtype=str)
    df["_csv_row"] = range(2, len(df) + 2)

    rev = pd.read_csv(review_csv, dtype=str)
    lines_to_delete: set[int] = set()

    for _, r in rev.iterrows():
        line = int(r["csv_data_line"])
        action = str(r["primary_action"])
        mok = r.get("manual_ok", "")
        if action in ("remove_row_headline_heuristic", "remove_row_standalone_pagination"):
            if _manual_explicit_yes(mok):
                lines_to_delete.add(line)

    lines_to_strip: dict[int, str] = {}
    for _, r in rev.iterrows():
        line = int(r["csv_data_line"])
        if line in lines_to_delete:
            continue
        flags = _cell_str(r.get("all_flags", ""))
        if "strip_pagination_suffix" not in flags:
            continue
        action = str(r["primary_action"])
        mok = r.get("manual_ok", "")
        # Strip-only rows: allow manual no to skip (rare). Headline+strip: keep verse => still strip Pages:.
        if action == "strip_pagination_suffix" and _manual_explicit_no(mok):
            continue
        lines_to_strip[line] = str(r["text_after_pagination_strip"])

    stripped_n = 0
    for idx in df.index:
        line = int(df.at[idx, "_csv_row"])
        if line in lines_to_strip:
            df.at[idx, "text"] = lines_to_strip[line]
            stripped_n += 1

    keep_mask = ~df["_csv_row"].isin(lines_to_delete)
    removed_n = int((~keep_mask).sum())
    df_out = df.loc[keep_mask].drop(columns=["_csv_row"]).copy()

    df_out.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"Applied review from {review_csv}")
    print(f"  Pagination strips applied: {stripped_n}")
    print(f"  Rows removed (manual_ok=yes): {removed_n}")
    print(f"  Rows in bible.csv now: {len(df_out)}")
    print("Next: rebuild lemmatized CSV, BM25 index, and embeddings for DK corpus.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove DK editorial headlines from bible.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Heuristic-only rewrite of bible.csv (destructive). Ignores manual_ok in review CSVs.",
    )
    parser.add_argument(
        "--apply-reviewed",
        action="store_true",
        help="Rewrite bible.csv using candidates_all.csv (manual_ok for removals; default strip for pagination).",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=None,
        help="Path to reviewed CSV for --apply-reviewed (default: data/dk/review_cleanup/candidates_all.csv)",
    )
    parser.add_argument(
        "--extract-review",
        action="store_true",
        help="Write review CSVs under data/dk/review_cleanup/; do not modify bible.csv",
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=None,
        help="Override output directory for --extract-review (default: data/dk/review_cleanup)",
    )
    args = parser.parse_args()

    if args.extract_review:
        run_extract_review(args.review_dir or REVIEW_DIR)
        return

    if args.apply_reviewed:
        if args.apply or args.dry_run:
            raise SystemExit("Use either --apply-reviewed or --apply/--dry-run, not both.")
        rc = args.review_csv or (REVIEW_DIR / "candidates_all.csv")
        run_apply_reviewed(rc)
        return

    if not args.dry_run and not args.apply:
        parser.print_help()
        print(
            "\nNo action: use --extract-review (export lists), --dry-run (preview), "
            "--apply-reviewed (after filling manual_ok), or --apply (heuristic only)."
        )
        return

    df = pd.read_csv(CSV_PATH, dtype=str)
    total_before = len(df)
    print(f"Rows before: {total_before}")

    # Strip embedded "Pages:\n1\n2..." from verse text (affects ~78 rows — last verse per book page)
    embedded_pagination = df["text"].str.contains("\nPages:", na=False)
    print(f"  Rows with embedded pagination suffix: {embedded_pagination.sum()}")
    df.loc[embedded_pagination, "text"] = df.loc[embedded_pagination, "text"].apply(_strip_pagination)

    heading_mask = df["text"].apply(_is_chapter_heading)
    pagination_mask = df["text"].apply(_is_pagination_artifact)
    remove_mask = heading_mask | pagination_mask

    removed_headings = heading_mask.sum()
    removed_pagination = pagination_mask.sum()
    print(f"  Headline rows to remove: {removed_headings}")
    print(f"  Standalone pagination rows to remove: {removed_pagination}")
    print(f"  Rows after cleanup: {total_before - remove_mask.sum()}")

    if args.dry_run:
        print("\n--- DRY RUN: first 20 headline rows that would be removed ---")
        for _, row in df[heading_mask].head(20).iterrows():
            print(f"  {row['book'][:40]},{row['chapter']},{row['verse']}: {str(row['text'])[:80]}")
        if not args.apply:
            return

    if not args.apply:
        return

    df_clean = df[~remove_mask].copy()
    df_clean.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"\nSaved cleaned CSV ({len(df_clean)} rows) to {CSV_PATH}")
    print("Next step: rebuild lemmatized CSV, BM25 index, and embeddings for DK corpus.")


if __name__ == "__main__":
    main()
