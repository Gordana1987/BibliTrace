"""
Apply known structural/text fixes to Bakotić Psalms in bible.csv.

Run from backend/:
  python scripts/fix_psalms_bakotic.py
  python scripts/fix_psalms_bakotic.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "bakotic" / "bible.csv"
BOOK = "Псалми"


def _mask(df: pd.DataFrame, ps: int, verse: int | None = None) -> pd.Series:
    m = (df.book == BOOK) & (df.chapter == ps)
    if verse is not None:
        m &= df.verse == verse
    return m


def _bump_verses(df: pd.DataFrame, ps: int, from_verse: int, delta: int) -> None:
    """Add delta to verse numbers >= from_verse (high-to-low if delta > 0)."""
    idx = df.index[_mask(df, ps) & (df.verse >= from_verse)]
    ascending = delta < 0
    for i in df.loc[idx].sort_values("verse", ascending=ascending).index:
        df.at[i, "verse"] = int(df.at[i, "verse"]) + delta


def apply_fixes(df: pd.DataFrame) -> list[str]:
    log: list[str] = []

    # Latin
    if _mask(df, 119, 116).any():
        i = df.index[_mask(df, 119, 116)][0]
        if "h" in df.at[i, "text"]:
            df.at[i, "text"] = df.at[i, "text"].replace("h", "ћ")
            log.append("119:116 latin h → ћ")

    if _mask(df, 136, 11).any():
        i = df.index[_mask(df, 136, 11)][0]
        t = df.at[i, "text"]
        if "a" in t and "Изр" in t:
            df.at[i, "text"] = t.replace("Изрaиља", "Израиља")
            log.append("136:11 latin a → а")

    # Ps 6
    if _mask(df, 6, 8).any() and not _mask(df, 6, 10).any():
        i = df.index[_mask(df, 6, 8)][0]
        text = df.at[i, "text"]
        if "глас плача" in text and "молитву моју прими" in text:
            v8 = "Идите од мене ви који неправду чините, јер Господ чу глас плача мога."
            v9 = "Господ чу молитву моју, Господ молитву моју прими."
            df.at[i, "text"] = v8
            j = df.index[_mask(df, 6, 9)][0]
            df.at[j, "verse"] = 10
            df.loc[len(df)] = {"book": BOOK, "chapter": 6, "verse": 9, "text": v9}
            log.append("6: split 8→8+9, old 9→10")

    # Ps 17
    dup3 = df.index[_mask(df, 17, 3)].tolist()
    if len(dup3) == 2 and not _mask(df, 17, 15).any():
        i1, i2 = dup3[0], dup3[1]
        t1 = df.at[i1, "text"]
        cut = " Речи моје не иду преко мисли мојих."
        if cut in t1:
            df.at[i1, "text"] = t1.replace(cut, ".")
        for v in range(14, 3, -1):
            for idx in df.index[_mask(df, 17, v)]:
                if idx != i2:
                    df.at[idx, "verse"] = v + 1
        df.at[i2, "verse"] = 4
        log.append("17: dup 3→4; shift 4–14 to 5–15")

    # Ps 22
    dup5 = df.index[_mask(df, 22, 5)].tolist()
    if len(dup5) == 2 and not _mask(df, 22, 4).any():
        df.at[dup5[0], "verse"] = 4
        log.append("22: first dup 5→4")

    # Ps 54
    if _mask(df, 54, 8).any() and not _mask(df, 54, 4).any():
        for v in range(5, 9):
            for idx in df.index[_mask(df, 54, v)]:
                df.at[idx, "verse"] = v - 1
        log.append("54: 5–8 → 4–7")

    # Ps 57
    if _mask(df, 57, 12).any() and not _mask(df, 57, 11).any():
        df.at[df.index[_mask(df, 57, 12)][0], "verse"] = 11
        log.append("57: 12→11")

    # Ps 61
    if _mask(df, 61, 7).any() and not _mask(df, 61, 8).any():
        i = df.index[_mask(df, 61, 7)][0]
        text = df.at[i, "text"]
        if "испуњавајући" in text:
            df.at[i, "text"] = (
                "да довека пред Богом пребива! Дај да га чувају милост твоја "
                "и истина твоја."
            )
            df.loc[len(df)] = {
                "book": BOOK,
                "chapter": 61,
                "verse": 8,
                "text": "Тако ћу певати имену твојему свагда, испуњавајући свагда завете своје.",
            }
            log.append("61: split 7→7+8")

    # Ps 102 — use temp verses to avoid collisions when shifting down
    if _mask(df, 102, 29).any() and not _mask(df, 102, 5).any():
        for idx in df.index[_mask(df, 102) & (df.verse >= 6)]:
            df.at[idx, "verse"] = int(df.at[idx, "verse"]) + 1000
        for idx in df.index[_mask(df, 102) & (df.verse >= 1006)]:
            df.at[idx, "verse"] = int(df.at[idx, "verse"]) - 1001
        log.append("102: 6–29 → 5–28")

    # Ps 109 — split 16 only; do not shift later verses
    if _mask(df, 109, 16).any() and not _mask(df, 109, 17).any():
        i = df.index[_mask(df, 109, 16)][0]
        if "Клетву је љубио" in df.at[i, "text"]:
            df.at[i, "text"] = (
                "Јер се он није сећао милосрђа, већ је гонио ништега и убогога, "
                "и тужном је срцу смрт тражио."
            )
            df.loc[len(df)] = {
                "book": BOOK,
                "chapter": 109,
                "verse": 17,
                "text": (
                    "Клетву је љубио, нека га и стигне; није марио за благослов, "
                    "нека и од њега побегне."
                ),
            }
            log.append("109: split 16→16+17")

    # Ps 115
    if _mask(df, 115, 17).any() and not _mask(df, 115, 18).any():
        i = df.index[_mask(df, 115, 17)][0]
        text = df.at[i, "text"]
        if "Алилуја" in text and "него ћемо" in text:
            a, b = text.split(" него ћемо", 1)
            df.at[i, "text"] = a.rstrip(".") + "."
            df.loc[len(df)] = {
                "book": BOOK,
                "chapter": 115,
                "verse": 18,
                "text": "Него ћемо" + b,
            }
            log.append("115: split 17→17+18")

    # Ps 140
    if _mask(df, 140, 14).any() and not _mask(df, 140, 13).any():
        df.at[df.index[_mask(df, 140, 14)][0], "verse"] = 13
        log.append("140: 14→13")

    return log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)
    before = len(df[df.book == BOOK])
    log = apply_fixes(df)
    after = len(df[df.book == BOOK])

    for line in log:
        print(line)
    print(f"Psalms rows: {before} → {after}")

    if not args.dry_run and log:
        df.to_csv(CSV_PATH, index=False)
        print(f"Updated {CSV_PATH}")
    elif args.dry_run:
        print("(dry run)")


if __name__ == "__main__":
    main()
