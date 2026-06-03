"""
Ingest Serbian SPC Bible from the EPUB offered by the Canadian diocese (Источник).

Source page (attribution, download button):
  https://istocnik.ca/sveto-pismo

The site serves the file via POST to /download-file with CSRF: the hidden field in the
form is not what the server checks — use the *cookie* varient_csrf_cookie as
varient_csrf_token in the POST body (same as a browser session).

Output (same columns as other corpora):
  backend/data/spc/bible.csv
    book, chapter, verse, text

Run from backend:
  python scripts/ingest_spc_epub.py --download   # fetch EPUB, then parse
  python scripts/ingest_spc_epub.py --epub path/to/file.epub
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

import ebooklib
from ebooklib import epub

BASE_DIR = Path(__file__).resolve().parents[1]
SPC_DIR = BASE_DIR / "data" / "spc"
DEFAULT_EPUB = SPC_DIR / "source.epub"
OUTPUT_CSV = SPC_DIR / "bible.csv"

SOURCE_PAGE = "https://istocnik.ca/sveto-pismo"
DOWNLOAD_POST = "https://istocnik.ca/download-file"

SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BibliTrace/1.0; research corpus)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "sr,en;q=0.9",
}

# <!-- #1: СЗ | 1 Мој - Прва књига Мојсијева ( 50 ) -->
BOOK_COMMENT_RE = re.compile(
    r"<!--\s*#\d+:\s*[^|]+\|\s*(.+?)\s*\(\s*\d+\s*\)\s*-->",
    re.DOTALL,
)

# Chapter rubric: "1 Мој 1. Title" or "Пс 3. Title"; skip "( 50 ГЛАВА )"
CHAPTER_H4_RE = re.compile(r"^(.+)\s+(\d+)\.\s+(.+)$")
SKIP_H4_RE = re.compile(r"^\(\s*\d+\s*ГЛАВА", re.IGNORECASE)

VERSE_MARK_RE = re.compile(r"\((\d+)\)")


def download_epub(dest: Path) -> None:
    """Fetch EPUB from istocnik.ca (session + CSRF cookie)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update(SESSION_HEADERS)
    r = s.get(SOURCE_PAGE, timeout=45)
    r.raise_for_status()
    token = s.cookies.get("varient_csrf_cookie")
    if not token:
        raise RuntimeError("No varient_csrf_cookie after GET; download may fail.")
    r2 = s.post(
        DOWNLOAD_POST,
        data={"varient_csrf_token": token, "file_id": "24"},
        timeout=120,
        headers={"Referer": SOURCE_PAGE, "Origin": "https://istocnik.ca"},
    )
    r2.raise_for_status()
    if not r2.content[:4] == b"PK\x03\x04":
        raise RuntimeError(
            "Response is not a ZIP/EPUB (expected PK header). "
            "The site layout or CSRF rules may have changed."
        )
    dest.write_bytes(r2.content)
    print(f"Wrote {len(r2.content)} bytes to {dest}")


def _extract_book_title_from_comment(html: str) -> str | None:
    m = BOOK_COMMENT_RE.search(html)
    return m.group(1).strip().replace("\n", " ") if m else None


def _split_verse_paragraph(blob: str) -> list[tuple[int, str]]:
    """Split '(1) foo (2) bar' into [(1, 'foo'), (2, 'bar'), ...]."""
    blob = re.sub(r"\s+", " ", blob).strip()
    if not VERSE_MARK_RE.search(blob):
        return []
    parts = VERSE_MARK_RE.split(blob)
    # parts: [lead, v1, text1, v2, text2, ...]
    out: list[tuple[int, str]] = []
    i = 1
    while i + 1 < len(parts):
        try:
            vn = int(parts[i])
        except ValueError:
            break
        txt = parts[i + 1].strip()
        if txt:
            out.append((vn, txt))
        i += 2
    return out


def parse_section_xhtml(html: str) -> list[dict]:
    rows: list[dict] = []
    book = _extract_book_title_from_comment(html)
    if not book:
        return rows

    soup = BeautifulSoup(html, "html.parser")
    body = soup.body
    if not body:
        return rows

    current_chapter: int | None = None

    for tag in body.find_all(["h4", "p"]):
        if tag.name == "h4":
            txt = tag.get_text(" ", strip=True)
            if not txt or SKIP_H4_RE.match(txt):
                continue
            m = CHAPTER_H4_RE.match(txt)
            if m:
                current_chapter = int(m.group(2))
            continue

        if tag.name != "p" or current_chapter is None:
            continue
        blob = tag.get_text(" ", strip=True)
        if not blob or not VERSE_MARK_RE.search(blob):
            continue
        for vn, vtxt in _split_verse_paragraph(blob):
            rows.append(
                {
                    "book": book,
                    "chapter": current_chapter,
                    "verse": vn,
                    "text": vtxt,
                }
            )

    return rows


def _spine_xhtml_hrefs(epub_path: Path) -> list[str]:
    """Return OEBPS-relative hrefs for document spine items in order."""
    with zipfile.ZipFile(epub_path) as zf:
        with zf.open("OEBPS/content.opf") as f:
            root = ET.parse(f).getroot()
    ns = {"opf": "http://www.idpf.org/2007/opf"}
    manifest: dict[str, str] = {}
    for item in root.findall(".//opf:manifest/opf:item", ns):
        iid = item.get("id")
        href = item.get("href")
        if iid and href:
            manifest[iid] = href.replace("\\", "/")
    hrefs: list[str] = []
    for ref in root.findall(".//opf:spine/opf:itemref", ns):
        iid = ref.get("idref")
        if iid and iid in manifest:
            hrefs.append(manifest[iid])
    return hrefs


def parse_epub_file(epub_path: Path) -> list[dict]:
    rows: list[dict] = []
    book = epub.read_epub(str(epub_path))
    by_name: dict[str, bytes] = {}
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            name = (item.get_name() or "").replace("\\", "/")
            by_name[name] = item.get_content()

    for href in _spine_xhtml_hrefs(epub_path):
        if not href.startswith("Text/"):
            continue
        if "Section0001_" not in href:
            continue
        if "CoverImage" in href or "SectionInfo" in href:
            continue
        raw = by_name.get(href)
        if raw is None:
            continue
        html = raw.decode("utf-8", errors="replace")
        rows.extend(parse_section_xhtml(html))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/spc/bible.csv from SPC EPUB")
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"Download EPUB from {SOURCE_PAGE} into --epub path, then ingest",
    )
    parser.add_argument(
        "--epub",
        type=Path,
        default=DEFAULT_EPUB,
        help=f"Path to .epub file (default: {DEFAULT_EPUB})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help=f"Output CSV (default: {OUTPUT_CSV})",
    )
    args = parser.parse_args()

    if args.download:
        download_epub(args.epub)

    if not args.epub.is_file():
        raise SystemExit(f"EPUB not found: {args.epub}. Use --download or pass --epub.")

    rows = parse_epub_file(args.epub)
    if not rows:
        raise SystemExit("No verses extracted; EPUB structure may have changed.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} verses to {args.output}")
    print(f"  Books (unique book field): {df['book'].nunique()}")


if __name__ == "__main__":
    main()
