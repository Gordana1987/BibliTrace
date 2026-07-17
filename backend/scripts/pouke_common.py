"""
Shared constants and HTML parsing for Поуке.орг Bible corpora.

Source: https://svetopismo.pouke.org/biblija.php
Verse text lives in div.def (usually in <p>); verse number in div.x > a[name].
Cross-refs, social links, and illustrations are outside the verse text.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

POUKE_BASE = "https://svetopismo.pouke.org/biblija.php"
DEFAULT_DELAY = 10.0  # robots.txt crawl-delay
CSV_COLUMNS = ["book", "chapter", "verse", "text"]

SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "sr,en;q=0.9",
    "Referer": "https://svetopismo.pouke.org/",
}


@dataclass(frozen=True)
class PoukeBook:
    num: int
    canonical: str
    chapters: int
    # Old DK CSV book label (for verse-count comparison only)
    legacy_dk: str


# Pouke book numbers 1–66; canonical names from audit_corpus_alignment.py
POUKE_BOOKS: list[PoukeBook] = [
    PoukeBook(1, "Постанак", 50, "Постанак"),
    PoukeBook(2, "Излазак", 40, "Излазак"),
    PoukeBook(3, "Левитска", 27, "Левитска"),
    PoukeBook(4, "Бројеви", 36, "Бројеви"),
    PoukeBook(5, "Поновљени закони", 34, "Поновљени закони"),
    PoukeBook(6, "Исус Навин", 24, "Исус Навин"),
    PoukeBook(7, "Судије", 21, "Судије"),
    PoukeBook(8, "Рута", 4, "Рута"),
    PoukeBook(9, "1. Самуилова", 31, "1. Самуилова"),
    PoukeBook(10, "2. Самуилова", 24, "2. Самуилова"),
    PoukeBook(11, "1. Краљевима", 22, "1. Краљевима"),
    PoukeBook(12, "2. Краљевима", 25, "2. Краљевима"),
    PoukeBook(13, "1. Летописа", 29, "1. Летописа"),
    PoukeBook(14, "2. Летописа", 36, "2. Летописа"),
    PoukeBook(15, "Јездра", 10, "Јездра"),
    PoukeBook(16, "Немија", 13, "Немија"),
    PoukeBook(17, "Јестира", 10, "Јестира"),
    PoukeBook(18, "Јов", 42, "Јов"),
    PoukeBook(19, "Псалми", 150, "Псалам"),
    PoukeBook(20, "Пословице", 31, "Пословице"),
    PoukeBook(21, "Проповедник", 12, "Проповедник"),
    PoukeBook(22, "Песма над песмама", 8, "Песма над песмама"),
    PoukeBook(23, "Исаија", 66, "Исаија"),
    PoukeBook(24, "Јеремија", 52, "Јеремија"),
    PoukeBook(25, "Тужбалице", 5, "Тужбалице"),
    PoukeBook(26, "Језекиљ", 48, "Језекиљ"),
    PoukeBook(27, "Данило", 12, "Данило"),
    PoukeBook(28, "Осија", 14, "Осија"),
    PoukeBook(29, "Јоило", 3, "Јоило"),
    PoukeBook(30, "Амос", 9, "Амос"),
    PoukeBook(31, "Авдија", 1, "Авдија"),
    PoukeBook(32, "Јона", 4, "Јона"),
    PoukeBook(33, "Михеј", 7, "Михеј"),
    PoukeBook(34, "Наум", 3, "Наум"),
    PoukeBook(35, "Авакум", 3, "Авакум"),
    PoukeBook(36, "Софонија", 3, "Софонија"),
    PoukeBook(37, "Агеј", 2, "Агеј"),
    PoukeBook(38, "Захарија", 14, "Захарија"),
    PoukeBook(39, "Малахија", 4, "Малахија"),
    PoukeBook(40, "Матеј", 28, "Матеј"),
    PoukeBook(41, "Марко", 16, "Марко"),
    PoukeBook(42, "Лука", 24, "Лука"),
    PoukeBook(43, "Јован", 21, "Јован"),
    PoukeBook(44, "Дела апостолска", 28, "Дела апостолска"),
    PoukeBook(45, "Римљанима", 16, "Римљанима"),
    PoukeBook(46, "1. Коринћанима", 16, "1. Коринћанима"),
    PoukeBook(47, "2. Коринћанима", 13, "2. Коринћанима"),
    PoukeBook(48, "Галатима", 6, "Галатима"),
    PoukeBook(49, "Ефешанима", 6, "Ефешанима"),
    PoukeBook(50, "Филипљанима", 4, "Филипљанима"),
    PoukeBook(51, "Колошанима", 4, "Колошанима"),
    PoukeBook(52, "1. Солуњанима", 5, "1. Солуњанима"),
    PoukeBook(53, "2. Солуњанима", 3, "2. Солуњанима"),
    PoukeBook(54, "1. Тимотеју", 6, "1. Тимотеју"),
    PoukeBook(55, "2. Тимотеју", 4, "2. Тимотеју"),
    PoukeBook(56, "Титу", 3, "Титу"),
    PoukeBook(57, "Филимону", 1, "Филимону"),
    PoukeBook(58, "Јеврејима", 13, "Јеврејима"),
    PoukeBook(59, "Јаковљева", 5, "Јаковљева"),
    PoukeBook(60, "1. Петрова", 5, "1. Петрова"),
    PoukeBook(61, "2. Петрова", 3, "2. Петрова"),
    PoukeBook(62, "1. Јованова", 5, "1. Јованова"),
    PoukeBook(63, "2. Јованова", 1, "2. Јованова"),
    PoukeBook(64, "3. Јованова", 1, "3. Јованова"),
    PoukeBook(65, "Јудина", 1, "Јудина"),
    PoukeBook(66, "Откривење", 22, "Откривење"),
]

POUKE_BOOK_BY_NUM = {b.num: b for b in POUKE_BOOKS}
CANONICAL_BOOK_ORDER = {b.canonical: b.num for b in POUKE_BOOKS}

CORPUS_LANG: dict[str, str] = {
    "dk": "ijekav",
    "dk_ekav": "ekavski",
    "spc": "sinod",
}


def chapter_url(lang: str, book_num: int, chapter: int) -> str:
    return f"{POUKE_BASE}?lang={lang}&book={book_num}&chap={chapter}"


def raw_chapter_path(raw_dir: Any, book_num: int, chapter: int) -> Any:
    return raw_dir / f"book_{book_num:02d}" / f"chapter_{chapter:03d}.html"


_DEF_BLOCK_RE = re.compile(
    r"""<div class=['"]def['"]>.*?(?=<div class=['"]y['"]>|</div><br)""",
    re.S,
)


def validate_raw_chapter_html(html: str) -> list[str]:
    """
    Return problem codes if downloaded HTML is not safe to parse.
    Run this before saving or parsing — catches Cloudflare truncation/injection.
    """
    issues: list[str] = []
    if not html or len(html) < 1000:
        issues.append("too_short")
    if "<div class='x'>" not in html and 'class="x"' not in html:
        issues.append("no_verse_markers")
    if "</body>" not in html and "footernav" not in html:
        issues.append("incomplete_page")

    def_blocks = _DEF_BLOCK_RE.findall(html)
    for block in def_blocks:
        if re.search(r"<script", block, re.I):
            issues.append("script_in_verse_block")
            break
        if "\ufffd" in block:
            issues.append("unicode_replacement_char")
            break
        if "<p>" in block and "</p>" not in block:
            issues.append("unclosed_verse_paragraph")
            break

    return list(dict.fromkeys(issues))  # preserve order, dedupe


_WHITESPACE_RUN_RE = re.compile(r"[ \t]+")


def _trim_verse_edges(text: str) -> str:
    """Strip \\r, collapse runs of spaces/tabs, outer trim — no word or punctuation edits."""
    text = text.replace("\r", "")
    text = _WHITESPACE_RUN_RE.sub(" ", text)
    return text.strip()


def _text_from_paragraph(p: Tag) -> str:
    """Concatenate paragraph strings; preserves punctuation and casing from source."""
    parts: list[str] = []
    for child in p.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name == "br":
                parts.append("\n")
            else:
                parts.append(child.get_text())
    return _trim_verse_edges("".join(parts))


def extract_verse_text(def_div: Tag) -> tuple[str, list[str]]:
    """
    Extract verse body from div.def.
    Returns (text, warnings) — warnings flag cases needing human review.
    """
    warnings: list[str] = []
    node = copy.copy(def_div)

    for ill in node.find_all("a", class_="ill-a"):
        ill.decompose()

    paragraphs = node.find_all("p")
    if paragraphs:
        if len(paragraphs) > 1:
            warnings.append("multiple_p_tags")
        text = "\n".join(_text_from_paragraph(p) for p in paragraphs if _text_from_paragraph(p))
        text = _trim_verse_edges(text)
    else:
        warnings.append("no_p_tag")
        text = _trim_verse_edges(node.get_text())
        if def_div.find("a", class_="ill-a"):
            warnings.append("had_illustration")

    if not text:
        warnings.append("empty_text")

    return text, warnings


_VERSE_NUM_RE = re.compile(r"^\d+\s*$")


def parse_chapter_html(html: str, *, book_name: str, chapter: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parse one chapter page into verse rows and parse warnings.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for x_div in soup.find_all("div", class_="x"):
        anchor = x_div.find("a", attrs={"name": True})
        if not anchor:
            continue
        name = anchor.get("name", "").strip()
        if not _VERSE_NUM_RE.match(name):
            issues.append(
                {
                    "book": book_name,
                    "chapter": chapter,
                    "issue": "bad_verse_anchor",
                    "detail": repr(name),
                }
            )
            continue
        verse_num = int(name)

        def_div = x_div.find_next_sibling("div", class_="def")
        if not def_div:
            issues.append(
                {
                    "book": book_name,
                    "chapter": chapter,
                    "verse": verse_num,
                    "issue": "missing_def_div",
                }
            )
            continue

        text, warnings = extract_verse_text(def_div)
        if warnings:
            issues.append(
                {
                    "book": book_name,
                    "chapter": chapter,
                    "verse": verse_num,
                    "issue": "parse_warning",
                    "warnings": warnings,
                    "text_preview": text[:120] if text else "",
                }
            )

        rows.append(
            {
                "book": book_name,
                "chapter": chapter,
                "verse": verse_num,
                "text": text,
            }
        )

    rows.sort(key=lambda r: (r["chapter"], r["verse"]))

    verses = [r["verse"] for r in rows]
    if verses != sorted(set(verses)):
        issues.append(
            {
                "book": book_name,
                "chapter": chapter,
                "issue": "duplicate_verse_numbers",
                "verses": verses,
            }
        )
    if verses and verses != list(range(1, len(verses) + 1)):
        issues.append(
            {
                "book": book_name,
                "chapter": chapter,
                "issue": "verse_number_gap",
                "verses": verses,
            }
        )

    return rows, issues


def filter_books(
    books: list[PoukeBook],
    *,
    only_nums: list[int] | None = None,
    from_num: int | None = None,
) -> list[PoukeBook]:
    out = books
    if only_nums:
        want = set(only_nums)
        out = [b for b in out if b.num in want]
    if from_num is not None:
        out = [b for b in out if b.num >= from_num]
    return out
