"""
Scrape the Bakotić Serbian Bible (OT + NT, Ekavian) from Serbian Wikisource.

INACTIVE in BibliTrace pipeline (2026-06): data/ kept for reference; app searches dk + spc only.
See config.ACTIVE_CORPORA. Script remains for manual maintenance if needed.

Source TOC:
  https://sr.wikisource.org/wiki/Библија_(Бакотић)

Each book is one Wikisource page. Verses live in <p> blocks; chapters in
<h3> as "Глава N." or "Псалам N." (Psalms). Internal newlines inside a
<p> are collapsed — one <p> = one verse row.

Output: backend/data/bakotic/bible.csv  (book, chapter, verse, text)

Run from backend/:
  python scripts/scrape_bible_bakotic.py
  python scripts/scrape_bible_bakotic.py --book Јеремија --output data/bakotic/jeremiah_test.csv
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://sr.wikisource.org"
TOC_URL = BASE_URL + "/sr-ec/Библија_(Бакотић)"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
BAKOTIC_DIR = DATA_DIR / "bakotic"
OUTPUT_PATH = BAKOTIC_DIR / "bible.csv"

DELAY_SECONDS = 1.5

BLOCK_TAGS = ("p", "h2", "h3", "h4")

CHAPTER_RE = re.compile(r"^\s*Глава\s+(\d+)\.?\s*$")
PSALM_RE = re.compile(r"^\s*Псалам\s+(\d+)\.?\s*$")
VERSE_RE = re.compile(r"^\s*(\d+)[\.\)]?\s+(.*\S)\s*$", re.DOTALL)

# Wikisource footnote refs flattened to "[ 1 ]" in verse text
FOOTNOTE_RE = re.compile(r"\s*\[\s*\d+\s*\]")
TRAILING_FOOTNOTE_RE = re.compile(r"\s*\.?\s*\[\s*$")

# Psalm superscriptions (Управитељу..., Псалам Давидов., Молитва..., etc.)
RUBRIC_PHRASE_RE = re.compile(
    r"^(?:"
    r"Управитељу збора певача[^.]*\.|"
    r"Псалам(?:\s+хвале)?[^.]*\.|"
    r"Песма[^.]*\.|"
    r"Молитва[^.]*\.|"
    r"Химна[^.]*\.|"
    r"Плачна песма[^.]*\.|"
    r"Поучење[^.]*\.|"
    r"(?:На гитару|по гуслама|за суботу)[^.]*\.|"
    r"(?:Једутуну|Хормашу|Микуру|Саломонов|Давидов)\.|"
    r"По мотиву[^.]*\.|"
    r"Уз (?:инструменте|фруле|харфу)[^.]*\.|"
    r"За (?:Гитит|харфу|спомен)[^.]*\.|"
    r"\[1\]\s*Алилуја[^.]*\."
    r")\s*",
    re.IGNORECASE,
)

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


def _block_text(tag) -> str:
    """Full text of a block element; collapse internal whitespace."""
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True))


def _strip_wikisource_footnotes(text: str) -> str:
    """Remove Wikisource footnote markers like '[ 1 ]' from verse text."""
    text = FOOTNOTE_RE.sub("", text)
    text = TRAILING_FOOTNOTE_RE.sub("", text)
    for punct in ",.;:":
        text = re.sub(rf"\s+{re.escape(punct)}", punct, text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"(\.)\s*\.+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_verse_text(text: str) -> str:
    return _strip_wikisource_footnotes(text)


def _psalm_paragraph_blob(tag) -> str:
    """
    Psalm <p> text with <i> superscriptions removed.

    Wikisource puts rubrics in italics; they are not part of any verse.
    """
    clone = BeautifulSoup(str(tag), "html.parser").find(tag.name)
    if clone is None:
        return _block_text(tag)
    for italic in clone.find_all("i"):
        italic.decompose()
    for sup in clone.find_all("sup", class_="reference"):
        sup.decompose()
    return re.sub(r"\s+", " ", clone.get_text(" ", strip=True))


def _strip_rubric_prefix(text: str) -> str:
    """Remove leading superscription phrases from verse text."""
    text = re.sub(r"\s+", " ", text).strip()
    while text:
        m = RUBRIC_PHRASE_RE.match(text)
        if not m:
            break
        text = text[m.end() :].strip()
    return text


def _is_rubric_only(text: str) -> bool:
    """True when verse text is only a superscription (no poetry)."""
    stripped = _strip_rubric_prefix(text)
    if not stripped:
        return True
    # Residual short attribution lines without prayer/poetry vocabulary.
    if len(stripped) < 80 and not re.search(
        r"Господ|Бог|Боже|човек|народ|срц|душ|земљ|небо|јер |ти |ја |ми |те ",
        stripped,
        re.IGNORECASE,
    ):
        return bool(
            re.match(
                r"^(?:Управитељу|Псалам|Песма|Молитва|Химна|Давидов|Саломонов)",
                stripped,
                re.IGNORECASE,
            )
        )
    return False


def _normalize_psalm_chapter_verses(
    pairs: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """
    Strip rubric prefixes; drop title-only verses; renumber when a verse is removed.

    Aligns with DK/SPC (poetry starts at v1) when the source numbers the title as v1.
    """
    cleaned: list[tuple[int, str]] = []
    dropped = False
    for verse_num, raw_text in pairs:
        text = _strip_rubric_prefix(raw_text)
        if not text or _is_rubric_only(text):
            dropped = True
            continue
        cleaned.append((verse_num, _clean_verse_text(text)))

    if not cleaned:
        return []
    if dropped:
        return [(i + 1, text) for i, (_, text) in enumerate(cleaned)]
    return cleaned


def _split_psalm_paragraph(blob: str) -> list[tuple[int, str]]:
    """
    Split a Psalms <p> into (verse_num, text) pairs.

    Wikisource often packs many verses in one paragraph:
      "1 Чуј, народе... 2 Отварам уста..."
    Rubric-only paragraphs (no verse numbers) return [].
    """
    blob = _strip_wikisource_footnotes(re.sub(r"\s+", " ", blob).strip())
    if not blob:
        return []

    m = VERSE_RE.match(blob)
    if m and not re.search(r"(?<!\d)(\d+)\s+\S", m.group(2)):
        return [(int(m.group(1)), _clean_verse_text(m.group(2)))]

    parts = re.split(r"(?<!\d)(\d+)\s+", blob)
    if len(parts) >= 3:
        out: list[tuple[int, str]] = []
        i = 1
        while i + 1 < len(parts):
            vn = int(parts[i])
            txt = _clean_verse_text(parts[i + 1])
            if txt:
                out.append((vn, txt))
            i += 2
        return out

    if m:
        return [(int(m.group(1)), _clean_verse_text(m.group(2)))]
    return []


def parse_book(html: str, book_name: str) -> list[dict]:
    """
    Parse one Bakotić book page into verse rows.

    Walk <p> and heading blocks in document order. Chapter headings set
    current chapter. Normal books: one <p> starting with a verse number = one row.
    Psalms: also split paragraphs that contain inline "N text" verse markers.
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", id="mw-content-text") or soup
    blocks = content.find_all(BLOCK_TAGS)

    has_chapters = any(
        CHAPTER_RE.match(_block_text(tag)) or PSALM_RE.match(_block_text(tag))
        for tag in blocks
    )

    rows: list[dict] = []
    current_chapter: int | None = None if has_chapters else 1
    psalm_chapter_pairs: list[tuple[int, str]] = []

    def flush_psalm_chapter(chapter: int | None) -> None:
        if chapter is None or not psalm_chapter_pairs:
            return
        for verse_num, verse_text in _normalize_psalm_chapter_verses(psalm_chapter_pairs):
            rows.append(
                {
                    "book": book_name,
                    "chapter": chapter,
                    "verse": verse_num,
                    "text": verse_text,
                }
            )

    for tag in blocks:
        text = _block_text(tag)
        if not text:
            continue

        mch = CHAPTER_RE.match(text) or PSALM_RE.match(text)
        if mch:
            if book_name == "Псалми":
                flush_psalm_chapter(current_chapter)
                psalm_chapter_pairs = []
            current_chapter = int(mch.group(1))
            continue

        if tag.name != "p" or current_chapter is None:
            continue

        if book_name == "Псалми":
            blob = _psalm_paragraph_blob(tag)
            psalm_chapter_pairs.extend(_split_psalm_paragraph(blob))
            continue

        m = VERSE_RE.match(text)
        verse_pairs = (
            [(int(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip())]
            if m and m.group(2).strip()
            else []
        )
        for verse_num, verse_text in verse_pairs:
            verse_text = _clean_verse_text(verse_text)
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

    if book_name == "Псалми":
        flush_psalm_chapter(current_chapter)
    return rows


def discover_book_links(session: requests.Session) -> list[tuple[str, str]]:
    print(f"Fetching TOC from {TOC_URL} ...")
    toc_html = fetch(TOC_URL, session)
    toc_soup = BeautifulSoup(toc_html, "html.parser")

    book_links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in toc_soup.find_all("a", href=True):
        href = a["href"]
        if "/wiki/%D0%91%D0%B8%D0%B1%D0%BB%D0%B8%D1%98%D0%B0_(%D0%91%D0%B0%D0%BA%D0%BE%D1%82%D0%B8%D1%9B)_:_" not in href:
            continue
        full_url = build_absolute_url(href)
        title = a.get_text(strip=True)
        if not title or full_url in seen:
            continue
        seen.add(full_url)
        book_links.append((title, full_url))
    return book_links


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Bakotić Bible from Wikisource.")
    parser.add_argument(
        "--book",
        help="Scrape only this book (exact title from TOC, e.g. Јеремија).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output CSV path (default: {OUTPUT_PATH}).",
    )
    args = parser.parse_args()

    BAKOTIC_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    book_links = discover_book_links(session)

    if not book_links:
        print("No Bakotić book links found on TOC page; aborting.")
        return

    if args.book:
        filtered = [(t, u) for t, u in book_links if t == args.book]
        if not filtered:
            print(f"Book not found: {args.book!r}")
            print("Available:", ", ".join(t for t, _ in book_links[:10]), "...")
            return
        book_links = filtered

    print(f"Scraping {len(book_links)} book(s) to {args.output} ...")

    all_rows: list[dict] = []
    for idx, (book, url) in enumerate(book_links, start=1):
        print(f"[{idx}/{len(book_links)}] {book} -> {url}")
        try:
            html = fetch(url, session)
            rows = parse_book(html, book)
            all_rows.extend(rows)
            print(f"  -> {len(rows)} verses")
        except Exception as e:
            print(f"  -> ERROR for {book}: {e}")
        if idx < len(book_links):
            time.sleep(DELAY_SECONDS)

    if not all_rows:
        print("No verses scraped; nothing to write.")
        return

    df = pd.DataFrame(all_rows, columns=["book", "chapter", "verse", "text"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved {len(df)} verses to {args.output}")


if __name__ == "__main__":
    main()
