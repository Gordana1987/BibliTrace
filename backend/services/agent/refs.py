"""Bible reference parsing + citation allowlist filter for ask-agent answers."""

from __future__ import annotations

import re

# Shared with agent eval — keep product and metrics on the same extractor.
REF_PATTERN = re.compile(
    r"(?:Мк|Марко|Мат|Матеј|Лк|Лука|Јн|Јован|Дела|Рим|1\.\s*Кор|2\.\s*Кор|Гал|Еф|Фил|Кол|"
    r"1\.\s*Сол|2\.\s*Сол|1\.\s*Тим|2\.\s*Тим|Тит|Филим|Јевр|Јаков|1\.\s*Пет|2\.\s*Пет|"
    r"1\.\s*Јов|2\.\s*Јов|3\.\s*Јов|Јуд|Откр|Откривење|"
    r"Римљанима|1\.\s*Коринћанима|2\.\s*Коринћанима|Галатима|Ефешанима|Филипљанима|"
    r"Колошанима|1\.\s*Солуњанима|2\.\s*Солуњанима|1\.\s*Тимотеју|2\.\s*Тимотеју|"
    r"Филимону|Јеврејима|Јаковљева|1\.\s*Петрова|2\.\s*Петрова|"
    r"1\.\s*Јованова|2\.\s*Јованова|3\.\s*Јованова|Јудина|Дела апостолска)"
    r"\s+(\d{1,3})\s*[:\.,]\s*(\d{1,3})",
)

BOOK_NORM = {
    "Мк": "Марко",
    "Мат": "Матеј",
    "Лк": "Лука",
    "Јн": "Јован",
    "Дела": "Дела апостолска",
    "Рим": "Римљанима",
    "Еф": "Ефешанима",
    "Фил": "Филипљанима",
    "Кол": "Колошанима",
    "Јевр": "Јеврејима",
    "Јаков": "Јаковљева",
    "Откр": "Откривење",
}

RefKey = tuple[str, int, int]


def normalize_book(raw: str) -> str:
    s = raw.strip()
    return BOOK_NORM.get(s, s)


def _match_to_ref(m: re.Match[str]) -> RefKey:
    full = m.group(0)
    ch, vs = int(m.group(1)), int(m.group(2))
    book_part = full[: m.start(1) - m.start(0)].strip().rstrip(":., ")
    return (normalize_book(book_part), ch, vs)


def extract_refs_from_answer(answer: str) -> set[RefKey]:
    """Return normalized (book, chapter, verse) refs found in free text."""
    if not answer:
        return set()
    return {_match_to_ref(m) for m in REF_PATTERN.finditer(answer)}


def filter_answer_to_allowed_refs(
    answer: str,
    allowed: set[RefKey],
) -> tuple[str, list[RefKey]]:
    """
    Strip verse references in ``answer`` that are not in ``allowed`` (tool hits).

    Minimal citation grounding: fabricated refs cannot leave in the answer text.
    Does not rewrite claims that lack an explicit book/chapter:verse pattern.
    """
    if not answer:
        return "", []

    removed: list[RefKey] = []
    spans: list[tuple[int, int]] = []
    for m in REF_PATTERN.finditer(answer):
        key = _match_to_ref(m)
        if key not in allowed:
            removed.append(key)
            spans.append(m.span())

    if not spans:
        return answer, []

    parts: list[str] = []
    prev = 0
    for start, end in spans:
        parts.append(answer[prev:start])
        prev = end
    parts.append(answer[prev:])
    text = "".join(parts)

    # dangling connectors left when a ref was the only object (… u Рим … → … u .)
    text = re.sub(r"\b(у|u)\s+(и|i)\s*", " ", text)
    text = re.sub(r"\b(у|u|и|i|на|из)\s*\.", ".", text)
    text = re.sub(r" +,", ",", text)
    text = re.sub(r" +;", ";", text)
    text = re.sub(r" +\(", "(", text)
    text = re.sub(r" +\.", ".", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip(), removed


def format_removed_refs_message(removed: list[RefKey]) -> str:
    if not removed:
        return ""
    # stable unique order
    uniq = sorted(set(removed), key=lambda r: (r[0], r[1], r[2]))
    refs = ", ".join(f"{b} {c}:{v}" for b, c, v in uniq)
    return f"Уклоњене референце ван резултата алата: {refs}."
