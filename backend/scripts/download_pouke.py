"""
Download raw HTML chapters from Поуке.орг (no text transformation).

Saves: data/<corpus>/raw/book_NN/chapter_MMM.html

Run from backend/:
  python scripts/download_pouke.py --corpus dk
  python scripts/download_pouke.py --corpus dk --book 1 --max-chapters 1
  python scripts/download_pouke.py --corpus dk --resume
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pouke_common import (
    CORPUS_LANG,
    DEFAULT_DELAY,
    POUKE_BOOKS,
    SESSION_HEADERS,
    chapter_url,
    filter_books,
    raw_chapter_path,
    validate_raw_chapter_html,
)

BASE_DIR = Path(__file__).resolve().parents[1]


def load_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(x) for x in data.get("completed", [])}


def save_progress(path: Path, completed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"completed": sorted(completed)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_chapter_curl(url: str) -> str:
    """curl returns full pages; requests often gets Cloudflare-truncated responses."""
    ua = SESSION_HEADERS["User-Agent"]
    result = subprocess.run(
        ["curl", "-sL", "-A", ua, url],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl exit {result.returncode}: {result.stderr[:200]}")
    return result.stdout


def fetch_chapter(
    session: requests.Session,
    url: str,
    *,
    retries: int = 2,
    validate: bool = True,
    use_curl: bool = True,
) -> str:
    last_err: Exception | None = None
    last_issues: list[str] = []
    for attempt in range(retries):
        try:
            html = fetch_chapter_curl(url) if use_curl else _fetch_chapter_requests(session, url)
            if validate:
                issues = validate_raw_chapter_html(html)
                if issues:
                    last_issues = issues
                    if attempt < retries - 1:
                        wait = 5.0 * (attempt + 1)
                        print(
                            f"    invalid HTML ({', '.join(issues)}), retry {attempt + 1}/{retries} in {wait:.0f}s …",
                            flush=True,
                        )
                        time.sleep(wait)
                    continue
            return html
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                wait = 3.0 * (attempt + 1)
                print(f"    error: {e}; retry in {wait:.0f}s …", flush=True)
                time.sleep(wait)
    if last_issues:
        raise RuntimeError(f"Invalid HTML after {retries} attempts: {last_issues}")
    assert last_err is not None
    raise last_err


def _fetch_chapter_requests(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=60, headers=SESSION_HEADERS)
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    return r.text


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw Поуке.орг chapter HTML")
    parser.add_argument(
        "--corpus",
        choices=sorted(CORPUS_LANG),
        default="dk",
        help="Target corpus folder (default: dk)",
    )
    parser.add_argument("--book", type=int, action="append", dest="books", help="Pouke book number (1–66)")
    parser.add_argument("--from-book", type=int, help="Start at this book number (inclusive)")
    parser.add_argument("--max-chapters", type=int, default=0, help="Limit chapters per book (0 = all)")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between requests")
    parser.add_argument("--resume", action="store_true", help="Skip chapters already on disk")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file exists (still validates before save)",
    )
    parser.add_argument(
        "--redo-failed",
        action="store_true",
        help="Re-download chapters listed in data/<corpus>/raw_validation.json",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failed chapters and continue (do not abort whole run)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Fetch retries per chapter when HTML fails validation (default: 2; use manual save if still failing)",
    )
    args = parser.parse_args()

    lang = CORPUS_LANG[args.corpus]
    data_dir = BASE_DIR / "data" / args.corpus
    raw_dir = data_dir / "raw"
    progress_path = data_dir / "pouke_download_progress.json"
    failures_path = data_dir / "pouke_download_failures.json"

    redo_tasks: list[tuple[int, int]] = []
    if args.redo_failed:
        validation_path = data_dir / "raw_validation.json"
        if not validation_path.exists():
            print(f"Missing {validation_path} — run validate_pouke_raw.py first.")
            return
        for row in json.loads(validation_path.read_text(encoding="utf-8")):
            redo_tasks.append((int(row["book_num"]), int(row["chapter"])))

    books = filter_books(POUKE_BOOKS, only_nums=args.books, from_num=args.from_book)
    if not books and not redo_tasks:
        print("No books matched filters.")
        return

    completed = load_progress(progress_path) if args.resume else set()
    session = requests.Session()

    failures: list[dict] = []

    def download_one(book_num: int, chapter: int, label: str) -> bool:
        key = f"{book_num}:{chapter}"
        out_path = raw_chapter_path(raw_dir, book_num, chapter)
        url = chapter_url(lang, book_num, chapter)
        try:
            html = fetch_chapter(session, url, retries=args.retries)
            issues = validate_raw_chapter_html(html)
            if issues:
                raise RuntimeError(str(issues))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(html, encoding="utf-8")
            completed.add(key)
            save_progress(progress_path, completed)
            print(f"  OK {label} → {out_path.name} ({len(html)} bytes)", flush=True)
            return True
        except Exception as e:
            msg = {"book_num": book_num, "chapter": chapter, "error": str(e)}
            failures.append(msg)
            print(f"  FAIL {label}: {e}", flush=True)
            return False

    print(f"Corpus: {args.corpus} (lang={lang})", flush=True)
    print(f"Raw dir: {raw_dir}", flush=True)

    if redo_tasks:
        print(f"Re-downloading {len(redo_tasks)} failed chapters, delay {args.delay}s", flush=True)
        for i, (book_num, chapter) in enumerate(redo_tasks, start=1):
            book = next(b for b in POUKE_BOOKS if b.num == book_num)
            label = f"#{book_num:02d} {book.canonical} {chapter} [{i}/{len(redo_tasks)}]"
            download_one(book_num, chapter, label)
            if i < len(redo_tasks):
                time.sleep(args.delay)
    else:
        total = sum(
            min(b.chapters, args.max_chapters) if args.max_chapters else b.chapters for b in books
        )
        done = 0
        print(f"Books: {len(books)}, chapters ≈ {total}, delay {args.delay}s", flush=True)

        for bi, book in enumerate(books, start=1):
            ch_max = book.chapters
            if args.max_chapters:
                ch_max = min(ch_max, args.max_chapters)

            print(f"[{bi}/{len(books)}] #{book.num} {book.canonical} ({ch_max} chapters)", flush=True)

            for chapter in range(1, ch_max + 1):
                key = f"{book.num}:{chapter}"
                out_path = raw_chapter_path(raw_dir, book.num, chapter)

                if (
                    not args.force
                    and args.resume
                    and out_path.exists()
                    and out_path.stat().st_size > 500
                    and not validate_raw_chapter_html(
                        out_path.read_text(encoding="utf-8", errors="replace")
                    )
                ):
                    done += 1
                    completed.add(key)
                    continue

                label = f"ch {chapter} [{done + 1}/{total}]"
                ok = download_one(book.num, chapter, label)
                if not ok and not args.continue_on_error:
                    failures_path.write_text(
                        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    raise SystemExit(1)
                done += 1

                if chapter < ch_max or bi < len(books):
                    time.sleep(args.delay)

    if failures:
        failures_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Failures ({len(failures)}) → {failures_path}", flush=True)

    print(f"Done.", flush=True)


if __name__ == "__main__":
    main()
