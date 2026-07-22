"""
Serbian script transliteration (Gaj Latin ↔ Cyrillic).

CLASSLA's sr models expect Latin script; BibliTrace corpora/queries are Cyrillic.
Use cyrillic_to_latin before CLASSLA, then keep Latin lemmas in indexes (matching
is script-consistent). Verse display text stays Cyrillic.
"""

from __future__ import annotations

# Order matters: digraphs before single letters; uppercase before lowercase.
_LATIN_TO_CYRILLIC: tuple[tuple[str, str], ...] = (
    ("DŽ", "Џ"),
    ("Dž", "Џ"),
    ("dž", "џ"),
    ("LJ", "Љ"),
    ("Lj", "Љ"),
    ("lj", "љ"),
    ("NJ", "Њ"),
    ("Nj", "Њ"),
    ("nj", "њ"),
    ("Đ", "Ђ"),
    ("đ", "ђ"),
    ("A", "А"),
    ("a", "а"),
    ("B", "Б"),
    ("b", "б"),
    ("V", "В"),
    ("v", "в"),
    ("G", "Г"),
    ("g", "г"),
    ("D", "Д"),
    ("d", "д"),
    ("E", "Е"),
    ("e", "е"),
    ("Ž", "Ж"),
    ("ž", "ж"),
    ("Z", "З"),
    ("z", "з"),
    ("I", "И"),
    ("i", "и"),
    ("J", "Ј"),
    ("j", "ј"),
    ("K", "К"),
    ("k", "к"),
    ("L", "Л"),
    ("l", "л"),
    ("M", "М"),
    ("m", "м"),
    ("N", "Н"),
    ("n", "н"),
    ("O", "О"),
    ("o", "о"),
    ("P", "П"),
    ("p", "п"),
    ("R", "Р"),
    ("r", "р"),
    ("S", "С"),
    ("s", "с"),
    ("T", "Т"),
    ("t", "т"),
    ("U", "У"),
    ("u", "у"),
    ("F", "Ф"),
    ("f", "ф"),
    ("H", "Х"),
    ("h", "х"),
    ("C", "Ц"),
    ("c", "ц"),
    ("Č", "Ч"),
    ("č", "ч"),
    ("Ć", "Ћ"),
    ("ć", "ћ"),
    ("Š", "Ш"),
    ("š", "ш"),
)

# Reverse map: Cyrillic digraph letters first (Љ/Њ/Џ), then singles.
_CYRILLIC_TO_LATIN: tuple[tuple[str, str], ...] = tuple(
    (cyr, lat) for lat, cyr in _LATIN_TO_CYRILLIC
)


def latin_to_cyrillic(text: str) -> str:
    """Convert Serbian Latin script to Cyrillic."""
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    out = text
    for lat, cyr in _LATIN_TO_CYRILLIC:
        out = out.replace(lat, cyr)
    return out


def cyrillic_to_latin(text: str) -> str:
    """Convert Serbian Cyrillic script to Latin (for CLASSLA and Latin-keyed tools)."""
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    out = text
    for cyr, lat in _CYRILLIC_TO_LATIN:
        out = out.replace(cyr, lat)
    return out
