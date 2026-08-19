"""App config and paths. Extend with env vars (e.g. DATA_DIR) when needed."""
from pathlib import Path

# Root of the backend package (for resolving data paths)
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"

# Temporary scope (2026-07): New Testament only; search each corpus separately (no merge).
ACTIVE_CORPORA: list[str] = ["dk", "spc"]

CORPUS_LABELS: dict[str, str] = {
    "dk": "Караџић (ијекав)",
    "spc": "СПЦ (ијекав)",
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

# Canonical NT order for exact/lemma result sorting (not alphabetical).
DK_NT_BOOK_ORDER: tuple[str, ...] = (
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
)

# UI / API hierarchy for NZ book filter (Pouke names; same for dk + spc).
NT_PAULINE_EPISTLES: tuple[str, ...] = (
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
)

NT_GENERAL_EPISTLES: tuple[str, ...] = (
    "Јаковљева",
    "1. Петрова",
    "2. Петрова",
    "1. Јованова",
    "2. Јованова",
    "3. Јованова",
    "Јудина",
)

NT_GOSPELS: tuple[str, ...] = ("Матеј", "Марко", "Лука", "Јован")


# Concept map — query expansion for semantic mode.
CONCEPT_MAP_PATH: Path = DATA_DIR / "concept" / "marko_concept_map_v1.json"

# Concept-search pagination (exact / lemma / semantic).
SEARCH_PAGE_SIZE: int = 20
# Absolute input ceiling (~one long verse); UI maxlength + API max_length.
SEARCH_MAX_TERM_CHARS: int = 200
# Hard word max for exact only (consecutive surface match).
SEARCH_MAX_TERM_TOKENS: int = 5
# Soft UI nudge for lemma / semantic (does not block search).
SEARCH_SOFT_WARN_TOKENS: int = 5
SEARCH_SOFT_WARN_MESSAGE: str = (
    "Ово личи на одломак, не појам — покушајте краћу фразу за боље резултате."
)
# Pure-dense semantic: max ranked hits (then paginate 20 + load more).
SEARCH_SEMANTIC_POOL: int = 100
# Live semantic encoder (Embedić e5-style; corpus built with passage: prefix).
SEARCH_EMBED_MODEL_ID: str = "djovak/embedic-large"
SEARCH_EMBED_INDEX_NAME: str = "embedic_large_nt_embeddings.joblib"
SEARCH_EMBED_QUERY_PREFIX: str = "query: "
SEARCH_EMBED_DOC_PREFIX: str = "passage: "


# Qwen query encoding for dense retrieval (literary analyze path / diagnostics).
# query = retrieval prompt (default); doc = same as verse encode; mean = L2-normalized average.
QUERY_ENCODE_MODES: tuple[str, ...] = ("query", "doc", "mean")
QUERY_ENCODE_MODE: str = "query"

# Surface synonym expansion before BM25 + dense encode (ON with CE for реч↔слово gate).
QUERY_EXPANSION_ENABLED: bool = True
QUERY_SYNONYMS: dict[str, list[str]] = {
    "реч": ["ријеч", "слово", "логос"],
    "ријеч": ["реч", "слово", "логос"],
    "слово": ["реч", "ријеч", "логос"],
    "логос": ["реч", "ријеч", "слово"],
}

# Cross-encoder final rerank: BM25 → embedding shortlist → CE top-k.
# Expansion stays OFF; CE only reshuffles verses already in the embedding pool.
CROSS_ENCODER_ENABLED: bool = True
CROSS_ENCODER_POOL: int = 50
CROSS_ENCODER_MODEL: str = "BAAI/bge-reranker-v2-m3"
