"""App config and paths. Extend with env vars (e.g. DATA_DIR) when needed."""
from pathlib import Path

# Root of the backend package (for resolving data paths)
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"

# Temporary scope (2026-07): New Testament only; search each corpus separately (no merge).
ACTIVE_CORPORA: list[str] = ["dk", "spc"]

CORPUS_LABELS: dict[str, str] = {
    "dk": "Даничић (ДК)",
    "spc": "СПЦ (НЗ)",
}

# Kept on disk; API/UI ignore these unless explicitly requested (then filtered out).
INACTIVE_CORPORA: list[str] = ["dk_ekav", "bakotic"]

# Restrict live retrieval to New Testament verses (Pouke short book names in bible.csv).
RESTRICT_TO_NEW_TESTAMENT: bool = True

# Exact Pouke book names for the 27 NT books (DK + SPC sinod share these labels).
DK_NT_BOOKS: frozenset[str] = frozenset(
    {
        "Матеј",
        "Марко",
        "Лука",
        "Јован",
        "Дела апостолска",
        "Римљанима",
        "1. Коринћанима",
        "2. Коринћанима",
        "Галатима",
        "Ефешанима",
        "Филипљанима",
        "Колошанима",
        "1. Солуњанима",
        "2. Солуњанима",
        "1. Тимотеју",
        "2. Тимотеју",
        "Титу",
        "Филимону",
        "Јеврејима",
        "Јаковљева",
        "1. Петрова",
        "2. Петрова",
        "1. Јованова",
        "2. Јованова",
        "3. Јованова",
        "Јудина",
        "Откривење",
    }
)
