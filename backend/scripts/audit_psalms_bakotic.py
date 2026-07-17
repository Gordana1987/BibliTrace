"""
Audit Bakotić Psalms vs DK (book «Псалам»).

Outputs data/bakotic/psalms_audit.csv with one row per issue:
  psalm, verse, issue_type, detail, suggested_action

Run from backend/:
  python scripts/audit_psalms_bakotic.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
BAK_PATH = BASE_DIR / "data" / "bakotic" / "bible.csv"
DK_PATH = BASE_DIR / "data" / "dk" / "bible.csv"
OUT_PATH = BASE_DIR / "data" / "bakotic" / "psalms_audit.csv"

BAK_BOOK = "Псалми"
DK_BOOK = "Псалам"

INLINE_RE = re.compile(r"(?<=[.!?;»\"])\s+(\d{1,3})\s+[А-ЯЂЈ]")
LATIN_RE = re.compile(r"[a-zA-Z]")
HYPHEN_RE = re.compile(r"[а-яА-ЯЂђЂјЈљЉњЊћЋџЏ]-[а-яА-ЯЂђЂјЈљЉњЊћЋџЏ]")
PLACE_PREFIXES = (
    "Вет", "Бет", "Киријат", "Тел", "Ен", "Фат", "Хот", "Асар", "Фохерет",
    "Керув", "Керен", "Овид", "Есион", "Гур", "Авел", "Тов", "Вен",
)


def _broken_hyphens(text: str) -> list[str]:
    out = []
    for m in HYPHEN_RE.finditer(text):
        w = m.group()
        if any(w.startswith(p) for p in PLACE_PREFIXES):
            continue
        parts = w.split("-")
        if any(len(p) <= 3 for p in parts):
            out.append(w)
    return out


def main() -> None:
    bak = pd.read_csv(BAK_PATH)
    dk = pd.read_csv(DK_PATH)
    b = bak[bak.book == BAK_BOOK].sort_values(["chapter", "verse"])
    d = dk[dk.book == DK_BOOK].sort_values(["chapter", "verse"])

    rows: list[dict] = []

    dup = b.groupby(["chapter", "verse"]).size()
    for (ps, v), count in sorted(dup[dup > 1].items()):
        rows.append({
            "psalm": ps,
            "verse": v,
            "issue_type": "duplicate",
            "detail": f"{count} rows at same ref",
            "suggested_action": "split or renumber second row",
        })

    for ps in range(1, 151):
        bs = set(b[b.chapter == ps].verse)
        ds = set(d[d.chapter == ps].verse)
        bc, dc = len(bs), len(ds)
        if bc != dc:
            rows.append({
                "psalm": ps,
                "verse": "",
                "issue_type": "count_diff",
                "detail": f"Bak {bc} vs DK {dc}",
                "suggested_action": "check alignment / splits",
            })
        for v in sorted(bs - ds):
            rows.append({
                "psalm": ps,
                "verse": v,
                "issue_type": "only_bak",
                "detail": "ref in Bakotić only",
                "suggested_action": "versification note or renumber",
            })
        for v in sorted(ds - bs):
            rows.append({
                "psalm": ps,
                "verse": v,
                "issue_type": "only_dk",
                "detail": "ref in DK only",
                "suggested_action": "split merged verse or add",
            })

    for _, r in b.iterrows():
        ps, v = int(r.chapter), int(r.verse)
        for m in INLINE_RE.finditer(r.text):
            rows.append({
                "psalm": ps,
                "verse": v,
                "issue_type": "merged_inline",
                "detail": f"inline verse {m.group(1)}",
                "suggested_action": "split at inline number",
            })
        if LATIN_RE.search(r.text):
            words = LATIN_RE.findall(r.text)
            rows.append({
                "psalm": ps,
                "verse": v,
                "issue_type": "latin_char",
                "detail": " ".join(words),
                "suggested_action": "replace with Cyrillic homoglyphs",
            })
        for w in _broken_hyphens(r.text):
            rows.append({
                "psalm": ps,
                "verse": v,
                "issue_type": "broken_hyphen",
                "detail": w,
                "suggested_action": "join syllables",
            })

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    n_ps = out["psalm"].nunique() if len(out) else 0
    print(f"Bak {len(b)} | DK {len(d)} | diff {len(b) - len(d)}")
    print(f"Issues: {len(out)} rows across {n_ps} psalms")
    print(f"Written {OUT_PATH}")
    if len(out):
        print(out.groupby("issue_type").size().to_string())


if __name__ == "__main__":
    main()
