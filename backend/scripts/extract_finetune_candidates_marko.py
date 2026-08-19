"""
Pilot: extract concept candidates from Gospel of Mark (DK + SPC).

No Embedić eval — lemma/POS/surface inventory + CLASSLA-split clusters only.

Run from backend/:
  python scripts/extract_finetune_candidates_marko.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from services.transliterate import cyrillic_to_latin, latin_to_cyrillic  # noqa: E402

BOOK = "Марко"
OUT = BASE_DIR / "data" / "concept" / "finetune_candidates_marko_v1.json"

# UPOS kept as content words
KEEP_UPOS = frozenset({"NOUN", "VERB", "ADJ", "PROPN"})
# PROPN whitelist: theological titles / concepts (not personal names / places)
PROPN_KEEP = frozenset(
    {
        "bog",
        "hrist",
        "hristos",
        "mesija",
        "duh",  # sometimes tagged PROPN in "Sveti Duh"
        "gospod",
        "otac",  # "Otac" as divine title — noisy but keep; filter later manually
    }
)

# Personal names / toponyms / frequent PROPN noise (Latin lemmas, lowercase)
PROPN_DROP = frozenset(
    {
        "isus",
        "petar",
        "jovan",
        "jakov",
        "andrej",
        "filip",
        "toma",
        "matej",
        "bartolomej",
        "simon",
        "juda",
        "marija",
        "marfa",
        "lavar",
        "jair",
        "pilat",
        "irod",
        "baraba",
        "pilat",
        "kaifa",
        "zaharija",
        "jelisaveta",
        "mojsije",
        "ilija",
        "david",
        "avram",
        "isaak",
        "jakov",  # patriarch overlap — still a name in narrative
        "galileja",
        "judeja",
        "jerusalim",
        "nazaret",
        "kaparnaum",
        "betanija",
        "betlehem",
        "sidon",
        "tir",
        "samarija",
        "jordan",
        "getsimanija",
        "golgota",
        "rim",
        "egipat",
        "misir",
        "izrailj",
        "izrael",
        "farisej",  # group name — keep? theological concept — KEEP via not listing
    }
)
# Remove farisej from drop — it's a theological category. Don't add to PROPN_DROP.

# Lemmas always dropped regardless of UPOS (function / light verbs / deixis)
LEMMA_DROP = frozenset(
    {
        "biti",
        "hteti",
        "moći",
        "morati",
        "imati",
        "nemati",
        "postati",
        "buditi",  # CLASSLA sometimes for "budite"
        "sebe",
        "se",
        "ja",
        "ti",
        "on",
        "ona",
        "ono",
        "mi",
        "vi",
        "oni",
        "one",
        "ona",
        "koji",
        "koja",
        "koje",
        "što",
        "šta",
        "taj",
        "ovaj",
        "onaj",
        "sav",
        "svaki",
        "neki",
        "nijedan",
        "jedan",  # often numeral/determiner noise
        "dva",
        "tri",
        "i",
        "a",
        "ali",
        "ili",
        "da",
        "ne",
        "li",
        "jer",
        "kad",
        "kada",
        "ako",
        "dok",
        "kao",
        "nego",
        "već",
        "još",
        "već",
        "samo",
        "tako",
        "ovde",
        "tu",
        "onde",
        "onda",
        "sada",
        "već",
        "u",
        "na",
        "od",
        "do",
        "za",
        "sa",
        "iz",
        "po",
        "k",
        "ka",
        "pre",
        "posle",
        "bez",
        "kroz",
        "oko",
        "među",
        "nad",
        "pod",
        "pri",
        "protiv",
        "zbog",
        "prema",
        "e",
        "o",
        "oh",
        "amen",
    }
)

# Minimum frequency (across both corpora combined) to keep a concept
MIN_FREQ = 2

# POS label for JSON (Serbian)
UPOS_TO_SR = {
    "NOUN": "imenica",
    "VERB": "glagol",
    "ADJ": "pridev",
    "PROPN": "vlastita_imenica",
}


def load_marko(corpus: str) -> pd.DataFrame:
    path = BASE_DIR / "data" / corpus / "bible_lemmatized.csv"
    df = pd.read_csv(path)
    m = df[df["book"].astype(str).str.strip() == BOOK].copy()
    return m.reset_index(drop=True)


def keep_token(lemma: str, upos: str) -> bool:
    lemma_l = lemma.lower().strip()
    if not lemma_l or lemma_l in LEMMA_DROP:
        return False
    if upos not in KEEP_UPOS:
        return False
    if upos == "PROPN":
        if lemma_l in PROPN_DROP:
            return False
        if lemma_l in PROPN_KEEP:
            return True
        # Drop other proper names / places by default
        return False
    # NOUN/VERB/ADJ
    if len(lemma_l) < 3 and lemma_l not in {"sin", "duh", "dan", "bog"}:
        return False
    return True


def extract_tokens(pipeline, df: pd.DataFrame, corpus: str):
    """Yield token dicts with lemma, form_cyr, upos, verse ref, corpus."""
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"CLASSLA {corpus}"):
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        if not text.strip():
            continue
        lat = cyrillic_to_latin(text)
        doc = pipeline(lat)
        ch, vs = int(row["chapter"]), int(row["verse"])
        for sent in doc.sentences:
            for w in sent.words:
                lemma = (w.lemma or w.text or "").strip()
                upos = (w.upos or "").strip()
                if not keep_token(lemma, upos):
                    continue
                surface_lat = (w.text or "").strip()
                if not surface_lat or re.fullmatch(r"\W+", surface_lat):
                    continue
                form_cyr = latin_to_cyrillic(surface_lat)
                rows.append(
                    {
                        "lemma": lemma.lower(),
                        "lemma_display": lemma,
                        "form": form_cyr,
                        "form_lat": surface_lat.lower(),
                        "upos": upos,
                        "pos": UPOS_TO_SR.get(upos, upos.lower()),
                        "corpus": corpus,
                        "chapter": ch,
                        "verse": vs,
                    }
                )
    return rows


def stem_key(lemma: str) -> str | None:
    """
    Aggressive stem for clustering CLASSLA-split families.
    Returns None if too short / unreliable.
    """
    w = lemma.lower()
    # Strip reflexive fluff already dropped
    suffixes = [
        "avanje",
        "ivanje",
        "evanje",
        "enje",
        "anje",
        "ost",
        "stvo",
        "štvo",
        "avati",
        "ivati",
        "evati",
        "ovati",
        "ati",
        "iti",
        "eti",
        "uti",
        "ći",
        "ao",
        "iti",
    ]
    # Special known roots first
    special = [
        ("blagosilj", "blagoslov"),
        ("blagoslov", "blagoslov"),
        ("spasav", "spas"),
        ("spas", "spas"),
        ("verov", "ver"),
        ("ver", "ver"),
        ("pokaj", "pokaj"),
        ("oprost", "oprost"),
        ("oprašt", "oprost"),
        ("prašt", "oprost"),
        ("molit", "mol"),
        ("molj", "mol"),
        ("propoved", "propoved"),
        ("propovijed", "propoved"),
        ("sudij", "sud"),
        ("suditi", "sud"),
        ("iscel", "iscel"),
        ("iscijel", "iscel"),
        ("iscjel", "iscel"),
        ("carstv", "carstv"),
        ("greh", "greh"),
        ("grijes", "greh"),
        ("vjer", "ver"),
        ("duhov", "duh"),
        ("krstit", "krst"),
        ("kršten", "krst"),
        ("učenic", "ucen"),
        ("učitelj", "ucen"),
        ("učiti", "ucen"),
        ("ljubav", "ljub"),
        ("ljubit", "ljub"),
        ("služ", "sluz"),
        ("moliti", "mol"),
    ]
    for pref, key in special:
        if w.startswith(pref):
            return key
    stem = w
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if stem.endswith(suf) and len(stem) - len(suf) >= 4:
                stem = stem[: -len(suf)]
                changed = True
                break
    if len(stem) < 4:
        return None
    return stem


def build_concepts(token_rows: list[dict]) -> list[dict]:
    # lemma -> aggregate
    agg: dict[str, dict] = {}
    for t in token_rows:
        lem = t["lemma"]
        if lem not in agg:
            agg[lem] = {
                "lemma": lem,
                "forms": set(),
                "pos_counts": defaultdict(int),
                "freq": 0,
                "corpora": set(),
                "verses": set(),
            }
        a = agg[lem]
        a["forms"].add(t["form"])
        a["pos_counts"][t["pos"]] += 1
        a["freq"] += 1
        a["corpora"].add(t["corpus"])
        a["verses"].add((t["chapter"], t["verse"]))

    concepts = []
    for lem, a in agg.items():
        if a["freq"] < MIN_FREQ:
            continue
        pos = max(a["pos_counts"].items(), key=lambda x: x[1])[0]
        corpora = a["corpora"]
        if corpora == {"dk", "spc"}:
            corp = "both"
        elif corpora == {"dk"}:
            corp = "dk"
        else:
            corp = "spc"
        verses = [
            {"book": BOOK, "chapter": ch, "verse": vs}
            for ch, vs in sorted(a["verses"])
        ]
        concepts.append(
            {
                "lemma": lem,
                "forms": sorted(a["forms"], key=lambda s: (-len(s), s))[:40],
                "pos": pos,
                "frequency": a["freq"],
                "corpus": corp,
                "n_verses": len(verses),
                "verses": verses,
            }
        )
    concepts.sort(key=lambda c: (-c["frequency"], c["lemma"]))
    return concepts


def build_clusters(concepts: list[dict]) -> list[dict]:
    by_lemma = {c["lemma"]: c for c in concepts}
    groups: dict[str, list[str]] = defaultdict(list)
    for lem in by_lemma:
        key = stem_key(lem)
        if key:
            groups[key].append(lem)

    # Known false friends: same stem key, unrelated sense
    FALSE_FRIENDS = {
        "ver": {"veriga"},  # chains/fetters ≠ faith
    }

    clusters = []
    for key, lems in sorted(groups.items(), key=lambda x: x[0]):
        drop = FALSE_FRIENDS.get(key, set())
        uniq = sorted(set(lems) - drop)
        if len(uniq) < 2:
            continue
        pos_mix = [by_lemma[l]["pos"] for l in uniq]
        note = (
            f"CLASSLA razdvaja u {len(uniq)} lema (stem≈{key}); "
            "kandidat za ručno povezivanje"
        )
        if len(set(pos_mix)) > 1:
            note += "; mešovite vrste reči (imenica↔glagol ili slično)"
        if drop:
            note += f"; uklonjeno false-friend: {sorted(drop)}"
        clusters.append(
            {
                "concept_cluster": key,
                "lemmas": uniq,
                "pos_mix": pos_mix,
                "frequencies": [by_lemma[l]["frequency"] for l in uniq],
                "note": note,
            }
        )
    clusters.sort(key=lambda c: (-len(c["lemmas"]), -sum(c["frequencies"])))
    return clusters


LIGHT_FOR_SHORTLIST = frozenset(
    {
        "reći",
        "doći",
        "ići",
        "govoriti",
        "kazati",
        "videti",
        "izići",
        "ući",
        "stati",
        "uzeti",
        "dati",
        "čuti",
        "znati",
        "gledati",
        "staviti",
        "metnuti",
        "pasti",
        "dignuti",
        "ustati",
        "sedeti",
        "leći",
        "trčati",
        "hodati",
        "pitati",
        "odgovoriti",
        "zvati",
        "dovesti",
        "odvesti",
        "poslati",
        "naći",
        "tražiti",
        "držati",
        "pustiti",
        "primiti",
        "primati",
        "drugi",
        "mnogi",
        "veliki",
        "mali",
        "prvi",
        "dan",
        "čas",
        "mesto",
        "čovek",
        "narod",
        "stvar",
    }
)


def build_review_shortlist(concepts: list[dict]) -> list[dict]:
    out = []
    for c in concepts:
        if c["frequency"] < 5:
            continue
        if c["lemma"] in LIGHT_FOR_SHORTLIST:
            continue
        if c["pos"] == "vlastita_imenica" and c["lemma"] not in {
            "bog",
            "hrist",
            "gospod",
            "duh",
        }:
            continue
        out.append(
            {
                "lemma": c["lemma"],
                "pos": c["pos"],
                "frequency": c["frequency"],
                "corpus": c["corpus"],
                "forms_sample": c["forms"][:8],
                "n_verses": c["n_verses"],
            }
        )
    return out


def main() -> None:
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from lemmatize_bible import build_pipeline, ensure_classla_model

    print("Loading Mark verses…")
    dk = load_marko("dk")
    spc = load_marko("spc")
    print(f"DK Marko={len(dk)}  SPC Marko={len(spc)}")

    ensure_classla_model("sr")
    pipeline = build_pipeline("sr", use_gpu=False)

    tokens = []
    tokens.extend(extract_tokens(pipeline, dk, "dk"))
    tokens.extend(extract_tokens(pipeline, spc, "spc"))
    print(f"Content tokens kept: {len(tokens)}")

    concepts = build_concepts(tokens)
    clusters = build_clusters(concepts)
    shortlist = build_review_shortlist(concepts)

    out = {
        "book": BOOK,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "source_csvs": [
            "data/dk/bible_lemmatized.csv",
            "data/spc/bible_lemmatized.csv",
        ],
        "method": (
            "CLASSLA tokenize+pos+lemma on Cyrillic→Latin verse text; "
            "surface forms latin_to_cyrillic(word.text); "
            f"keep UPOS={sorted(KEEP_UPOS)}; drop function lemmas + most PROPN; "
            f"min_frequency={MIN_FREQ} (token occurrences across dk+spc)"
        ),
        "n_concepts": len(concepts),
        "n_clusters": len(clusters),
        "n_review_shortlist": len(shortlist),
        "pilot_stats": {
            "dk_verses": len(dk),
            "spc_verses": len(spc),
            "content_token_rows": len(tokens),
            "concepts_freq_ge_2": len(concepts),
            "concepts_freq_ge_5": sum(1 for c in concepts if c["frequency"] >= 5),
            "concepts_freq_ge_10": sum(1 for c in concepts if c["frequency"] >= 10),
            "clusters_multi_lemma": len(clusters),
            "review_shortlist_n": len(shortlist),
            "pos_breakdown": {
                p: sum(1 for c in concepts if c["pos"] == p)
                for p in ("imenica", "glagol", "pridev", "vlastita_imenica")
            },
            "corpus_breakdown": {
                k: sum(1 for c in concepts if c["corpus"] == k)
                for k in ("both", "dk", "spc")
            },
            "nz_scale_estimate_rough": (
                f"~{len(concepts)} concepts in Mark; naive ×27≈{len(concepts)*27}, "
                f"realistic ~3–6× after cross-book dedup "
                f"({len(concepts)*3}–{len(concepts)*6})"
            ),
            "filter_notes": {
                "noise_level": (
                    "Full concepts (~inventory) still has narrative verbs; "
                    "use review_shortlist for manual theological pass."
                ),
                "mark_aspect_note": (
                    "In Mark, spas* mostly collapses to spasti; blagosiljati absent "
                    "(blagosloviti/blagosloven only). Fuller aspect splits appear at NZ scale."
                ),
            },
        },
        "review_shortlist": shortlist,
        "concepts": concepts,
        "clusters": clusters,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")
    print(f"n_concepts={out['n_concepts']}  n_clusters={out['n_clusters']}")
    print("pos:", out["pilot_stats"]["pos_breakdown"])
    print("corpus:", out["pilot_stats"]["corpus_breakdown"])
    print("\nTop 15 concepts:")
    for c in concepts[:15]:
        print(f"  {c['frequency']:4d}  {c['lemma']:<20} {c['pos']:<10} {c['corpus']}")
    print("\nTop clusters:")
    for cl in clusters[:20]:
        print(f"  {cl['concept_cluster']}: {cl['lemmas']}")


if __name__ == "__main__":
    main()
