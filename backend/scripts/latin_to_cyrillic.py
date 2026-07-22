"""
Serbian Latin → Cyrillic transliteration (Gaj's alphabet).

Used for JW sr-latn Bible ingest: we fetch Latin text and convert to Cyrillic
for BibliTrace (queries and corpora are Cyrillic-only).

Canonical implementation: services.transliterate (also provides cyrillic_to_latin).
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.transliterate import cyrillic_to_latin, latin_to_cyrillic  # noqa: E402

__all__ = ["latin_to_cyrillic", "cyrillic_to_latin"]
