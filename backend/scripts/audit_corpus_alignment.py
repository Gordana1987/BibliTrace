"""
Compare verse references across DK, Bakotić, and SPC (66-book Protestant canon).

Outputs:
  data/audit/corpus_chapter_diffs.csv   — chapters where max verse differs
  data/audit/corpus_missing_verses.csv — (ch, v) present in ≥1 corpus but not all
  data/audit/corpus_duplicates.csv    — duplicate book/chapter/verse within a corpus
  data/audit/corpus_book_summary.csv    — per-book totals and flags

Run from backend/:
  python scripts/audit_corpus_alignment.py
  python scripts/audit_corpus_alignment.py --book Постанак
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "audit"

# (canonical label, dk book, bakotic book, spc book)
BOOKS: list[tuple[str, str, str, str]] = [
    ("Постанак", "Постанак", "Постање", "1 Мој - Прва књига Мојсијева"),
    ("Излазак", "Излазак", "Излазак", "2 Мој - Друга књига Мојсијева"),
    ("Левитска", "Левитска", "Левитска", "3 Мој - Трећа књига Мојсијева"),
    ("Бројеви", "Бројеви", "Бројеви", "4 Мој - Четврта књига Мојсијева"),
    ("Поновљени закони", "Поновљени закони", "Поновљени закони", "5 Мој - Пета књига Мојсијева"),
    ("Исус Навин", "Исус Навин", "Исус Навин", "ИНав - Књига Исуса Навина"),
    ("Судије", "Судије", "Судије", "Суд - Књига о судијама"),
    ("Рута", "Рута", "Рута", "Рута - Књига о Рути"),
    ("1. Самуилова", "1. Самуилова", "Прва Књига Самуилова", "1 Сам - Прва књига Самуилова"),
    ("2. Самуилова", "2. Самуилова", "Друга Књига Самуилова", "2 Сам - Друга књига Самуилова"),
    ("1. Краљевима", "1. Краљевима", "Прва Књига о Краљевима", "1 Цар - Прва књига о царевима"),
    ("2. Краљевима", "2. Краљевима", "Друга Књига о Краљевима", "2 Цар - Друга књига о царевима"),
    ("1. Летописа", "1. Летописа", "Прва Књига Летописа", "1 Дн - Прва књига дневника"),
    ("2. Летописа", "2. Летописа", "Друга Књига Летописа", "2 Дн - Друга књига дневника"),
    ("Јездра", "Јездра", "Јездра", "1 Језд - Прва књига Јездрина"),
    ("Немија", "Немија", "Немија", "Нем - Књига Немијина"),
    ("Јестира", "Јестира", "Јестира", "Јест - Књига о Јестири"),
    ("Јов", "Јов", "Јов", "Јов - Књига о Јову"),
    ("Псалми", "Псалам", "Псалми", "Пс - Псалми Давидови"),
    ("Пословице", "Пословице", "Пословице", "ПрС - Приче Соломонове"),
    ("Проповедник", "Проповедник", "Проповедник", "Проп - Књига проповједникова"),
    ("Песма над песмама", "Песма над песмама", "Песма над песмама", "Пп - Пјесма над пјесмама"),
    ("Исаија", "Исаија", "Исаија", "Ис - Књига пророка Исаије"),
    ("Јеремија", "Јеремија", "Јеремија", "Јер - Књига пророка Јеремије"),
    ("Тужбалице", "Тужбалице", "Јеремијине Тужбалице", "ПлЈ - Плач Јеремијин"),
    ("Језекиљ", "Језекиљ", "Језекиљ", "Јез - Књига пророка Језекиља"),
    ("Данило", "Данило", "Данило", "Дан - Књига пророка Данила"),
    ("Осија", "Осија", "Осија", "Ос - Књига пророка Осије"),
    ("Јоило", "Јоило", "Јоиљ", "Јл - Књига пророка Јоила"),
    ("Амос", "Амос", "Амос", "Ам - Књига пророка Амоса"),
    ("Авдија", "Авдија", "Авдија", "Авд - Књига пророка Авдије"),
    ("Јона", "Јона", "Јона", "Јона - Књига пророка Јоне"),
    ("Михеј", "Михеј", "Михеј", "Мих - Књига пророка Михеја"),
    ("Наум", "Наум", "Наум", "Нм - Књига пророка Наума"),
    ("Авакум", "Авакум", "Авакум", "Авак - Књига пророка Авакума"),
    ("Софонија", "Софонија", "Софонија", "Соф - Књига пророка Софоније"),
    ("Агеј", "Агеј", "Агеј", "Аг - Књига пророка Агеја"),
    ("Захарија", "Захарија", "Захарија", "Зах - Књига пророка Захарије"),
    ("Малахија", "Малахија", "Малахија", "Мал - Књига пророка Малахије"),
    ("Матеј", "Матеј", "Јеванђеље по Матеју", "Мт - Свето Јеванђеље од Матеја"),
    ("Марко", "Марко", "Јеванђеље по Марку", "Мк - Свето Јеванђеље од Марка"),
    ("Лука", "Лука", "Јеванђеље по Луки", "Лк - Свето Јеванђеље од Луке"),
    ("Јован", "Јован", "Јеванђеље по Јовану", "Јн - Свето Јеванђеље од Јована"),
    ("Дела апостолска", "Дела апостолска", "Дела Апостолска", "Дап - Дјела Светих апостола"),
    ("Римљанима", "Римљанима", "Павлова посланица Римљанима", "Рим - Посланица Св. апостола Павла Римљанина"),
    ("1. Коринћанима", "1. Коринћанима", "Прва Павлова посланица Коринћанима", "1 Кор - Прва посланица Св. апостола Павла Коринћанима"),
    ("2. Коринћанима", "2. Коринћанима", "Друга Павлова посланица Коринћанима", "2 Кор - Друга посланица Св. апостола Павла Коринћанима"),
    ("Галатима", "Галатима", "Павлова посланица Галатима", "Гал - Посланица Св. апостола Павла Галатима"),
    ("Ефешанима", "Ефешанима", "Павлова посланица Ефесцима", "Еф - Посланица Св. апостола Павла Ефесцима"),
    ("Филипљанима", "Филипљанима", "Павлова посланица Филипљанима", "Флп - Посланица Св. апостола Павла Филипљанима"),
    ("Колошанима", "Колошанима", "Павлова посланица Колошанима", "Кол - Посланица Св. апостола Павла Колошанима"),
    ("1. Солуњанима", "1. Солуњанима", "Прва Павлова посланица Солуњанима", "1 Сол - Прва посланица Св. апостола Павла Солуњанима"),
    ("2. Солуњанима", "2. Солуњанима", "Друга Павлова посланица Солуњанима", "2 Сол - Друга посланица Св. апостола Павла Солуњанима"),
    ("1. Тимотеју", "1. Тимотеју", "Прва Павлова посланица Тимотеју", "1 Тим - Прва посланица Св. апостола Павла Тимотеју"),
    ("2. Тимотеју", "2. Тимотеју", "Друга Павлова посланица Тимотеју", "2 Тим - Друга посланица Св. апостола Павла Тимотеју"),
    ("Титу", "Титу", "Павлова посланица Титу", "Тит - Посланица Св. апостола Павла Титу"),
    ("Филимону", "Филимону", "Павлова посланица Филимону", "Флм - Посланица Св. апостола Павла Филимону"),
    ("Јеврејима", "Јеврејима", "Посланица Јеврејима", "Јев - Посланица Св. апостола Павла Јеврејима"),
    ("Јаковљева", "Јаковљева", "Јаковљева посланица", "Јак - Саборна посланица Св. апостола Јакова"),
    ("1. Петрова", "1. Петрова", "Прва Петрова посланица", "1 Пет - Прва саборна посланица Св. апостола Петра"),
    ("2. Петрова", "2. Петрова", "Друга Петрова посланица", "2 Пет - Друга саборна посланица Св. апостола Петра"),
    ("1. Јованова", "1. Јованова", "Прва Јованова посланица", "1 Јн - Прва саборна посланица Св. апостола Јована Богослова"),
    ("2. Јованова", "2. Јованова", "Друга Јованова посланица", "2 Јн - Друга саборна посланица Св. апостола Јована Богослова"),
    ("3. Јованова", "3. Јованова", "Трећа Јованова посланица", "3 Јн - Трећа саборна посланица Св. апостола Јована Богослова"),
    ("Јудина", "Јудина", "Јудина посланица", "Јуд - Саборна посланица Св. апостола Јуде (Јаковљевог)"),
    ("Откривење", "Откривење", "Јованово откривење", "Отк - Откривење Светога Јована Богослова"),
]


def _refs(df: pd.DataFrame) -> set[tuple[int, int]]:
    return {(int(r.chapter), int(r.verse)) for r in df.itertuples()}


def _duplicates(df: pd.DataFrame, corpus: str, label: str) -> list[dict]:
    dup = df.groupby(["chapter", "verse"]).size()
    rows = []
    for (ch, v), n in dup[dup > 1].items():
        rows.append({"knjiga": label, "corpus": corpus, "chapter": ch, "verse": v, "count": int(n)})
    return rows


def audit_book(
    label: str, dk_b: str, bak_b: str, spc_b: str, dk: pd.DataFrame, bak: pd.DataFrame, spc: pd.DataFrame
) -> tuple[list[dict], list[dict], list[dict], dict]:
    dk_sub = dk[dk.book == dk_b][["chapter", "verse"]]
    bak_sub = bak[bak.book == bak_b][["chapter", "verse"]]
    spc_sub = spc[spc.book == spc_b][["chapter", "verse"]]

    dk_refs, bak_refs, spc_refs = _refs(dk_sub), _refs(bak_sub), _refs(spc_sub)
    all_refs = dk_refs | bak_refs | spc_refs

    chapter_diffs = []
    missing = []
    for ch in sorted({c for c, _ in all_refs}):
        dk_max = max((v for c, v in dk_refs if c == ch), default=0)
        bak_max = max((v for c, v in bak_refs if c == ch), default=0)
        spc_max = max((v for c, v in spc_refs if c == ch), default=0)
        if len({dk_max, bak_max, spc_max} - {0}) > 1 and not (dk_max == bak_max == spc_max):
            chapter_diffs.append(
                {
                    "knjiga": label,
                    "chapter": ch,
                    "DK_max": dk_max,
                    "Bak_max": bak_max,
                    "SPC_max": spc_max,
                }
            )

    for ch, v in sorted(all_refs):
        in_dk, in_bak, in_spc = (ch, v) in dk_refs, (ch, v) in bak_refs, (ch, v) in spc_refs
        if in_dk and in_bak and in_spc:
            continue
        missing.append(
            {
                "knjiga": label,
                "chapter": ch,
                "verse": v,
                "DK": in_dk,
                "Bakotić": in_bak,
                "SPC": in_spc,
            }
        )

    dups = (
        _duplicates(dk_sub, "DK", label)
        + _duplicates(bak_sub, "Bakotić", label)
        + _duplicates(spc_sub, "SPC", label)
    )

    summary = {
        "knjiga": label,
        "DK_stihova": len(dk_sub),
        "Bak_stihova": len(bak_sub),
        "SPC_stihova": len(spc_sub),
        "DK−Bak": len(dk_sub) - len(bak_sub),
        "DK−SPC": len(dk_sub) - len(spc_sub),
        "poglavlja_razl_max": len(chapter_diffs),
        "nedostajuci_ref": len(missing),
        "duplikati": len(dups),
        "usklađeno": len(missing) == 0 and len(chapter_diffs) == 0 and len(dups) == 0
        and len(dk_sub) == len(bak_sub) == len(spc_sub),
    }
    return chapter_diffs, missing, dups, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DK / Bakotić / SPC verse alignment.")
    parser.add_argument("--book", help="Canonical book label, e.g. Постанак")
    args = parser.parse_args()

    dk = pd.read_csv(DATA_DIR / "dk" / "bible.csv")
    bak = pd.read_csv(DATA_DIR / "bakotic" / "bible.csv")
    spc = pd.read_csv(DATA_DIR / "spc" / "bible.csv")

    books = BOOKS
    if args.book:
        books = [b for b in BOOKS if b[0] == args.book]
        if not books:
            print(f"Unknown book: {args.book!r}")
            return

    all_ch, all_miss, all_dup, summaries = [], [], [], []
    for label, dk_b, bak_b, spc_b in books:
        ch, miss, dup, summ = audit_book(label, dk_b, bak_b, spc_b, dk, bak, spc)
        all_ch.extend(ch)
        all_miss.extend(miss)
        all_dup.extend(dup)
        summaries.append(summ)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUT_DIR / "corpus_book_summary.csv", index=False)
    pd.DataFrame(all_ch).to_csv(OUT_DIR / "corpus_chapter_diffs.csv", index=False)
    pd.DataFrame(all_miss).to_csv(OUT_DIR / "corpus_missing_verses.csv", index=False)
    pd.DataFrame(all_dup).to_csv(OUT_DIR / "corpus_duplicates.csv", index=False)

    ok = sum(1 for s in summaries if s["usklađeno"])
    print(f"Books audited: {len(summaries)}")
    print(f"Fully aligned (refs + no dups): {ok} / {len(summaries)}")
    print(f"Chapter max-verse diffs: {len(all_ch)}")
    print(f"Missing refs (not in all 3): {len(all_miss)}")
    print(f"Duplicate refs: {len(all_dup)}")
    print(f"Output: {OUT_DIR}/")


if __name__ == "__main__":
    main()
