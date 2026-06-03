"""
Serbian Latin → Cyrillic transliteration (Gaj's alphabet).

Used for JW sr-latn Bible ingest: we fetch Latin text and convert to Cyrillic
for BibliTrace (queries and corpora are Cyrillic-only).
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


def latin_to_cyrillic(text: str) -> str:
    """Convert Serbian Latin script to Cyrillic."""
    out = text
    for lat, cyr in _LATIN_TO_CYRILLIC:
        out = out.replace(lat, cyr)
    return out
