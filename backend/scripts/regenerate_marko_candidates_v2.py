"""
Regenerate Marko finetune candidates v2 with transparent keep/drop/borderline labels.

Reads v1 inventory (same 1020 lemmas) — does not re-run CLASSLA.
Writes data/concept/finetune_candidates_marko_v2.json

Run from backend/:
  python scripts/regenerate_marko_candidates_v2.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))

from extract_finetune_candidates_marko import (  # noqa: E402
    LIGHT_FOR_SHORTLIST,
    stem_key,
)

V1 = BASE_DIR / "data" / "concept" / "finetune_candidates_marko_v1.json"
OUT = BASE_DIR / "data" / "concept" / "finetune_candidates_marko_v2.json"

# Aspect-pair markers: imperfective -avati/-ivati vs perfective -ati/-iti
ASPECT_SUFFIX_PAIRS = (
    ("avati", "ati"),
    ("ivati", "iti"),
    ("evati", "eti"),
)

# Stem-key false friends (share stem heuristic but wrong sense)
FALSE_FRIENDS_BY_STEM: dict[str, dict[str, str]] = {
    "ver": {
        "veriga": "isti stem≈ver ali značenje 'lanac/okov', ne 'vera' — false friend"
    },
}

# PROPN keep set (theological)
PROPN_KEEP = frozenset({"bog", "hrist", "gospod", "duh"})


def verse_key(v: dict) -> tuple[int, int]:
    return (int(v["chapter"]), int(v["verse"]))


def classify_concept(c: dict) -> tuple[str, str | None]:
    """Return (status, drop_reason)."""
    freq = int(c.get("frequency") or c.get("freq") or 0)
    lemma = c["lemma"]
    pos = c["pos"]

    # Borderline: rare but visible tail (freq 2–4)
    if 2 <= freq <= 4:
        return "borderline", None

    # freq >= 5 from here
    if pos == "vlastita_imenica" and lemma not in PROPN_KEEP:
        return "dropped", "proper_noun"

    if lemma in LIGHT_FOR_SHORTLIST:
        return "dropped", "narrative_noise"

    return "keep", None


def join_basis_for(lemmas: list[str], pos_mix: list[str]) -> str:
    """Classify why lemmas were clustered."""
    # Aspect pair: two+ verbs, stems match after stripping aspectival suffixes
    verbs = [l for l, p in zip(lemmas, pos_mix) if p == "glagol"]
    if len(verbs) >= 2:
        norms = set()
        for v in verbs:
            n = v
            for long, short in ASPECT_SUFFIX_PAIRS:
                if n.endswith(long) and len(n) > len(long) + 3:
                    n = n[: -len(long)] + short
                    break
            norms.add(n)
        # Also treat opraštati/oprostiti/praštati via shared stem_key
        keys = {stem_key(v) for v in verbs}
        if len(norms) == 1 or (len(keys) == 1 and None not in keys):
            if any(
                v.endswith(suf)
                for v in verbs
                for suf in ("avati", "ivati", "evati", "ati", "iti")
            ):
                # Prefer aspect_pair when imperfective/perfective both present
                has_impf = any(v.endswith(("avati", "ivati", "evati")) for v in verbs)
                has_pf = any(
                    (v.endswith(("ati", "iti", "eti")) and not v.endswith(("avati", "ivati", "evati")))
                    for v in verbs
                )
                if has_impf and has_pf:
                    return "aspect_pair"

    # Manual special stems (from extract script special[] list)
    manual_stems = {
        "blagoslov",
        "spas",
        "ver",
        "pokaj",
        "oprost",
        "mol",
        "propoved",
        "iscel",
        "krst",
        "ucen",
        "ljub",
        "sluz",
        "greh",
        "carstv",
        "duh",
        "sud",
    }
    keys = {stem_key(l) for l in lemmas}
    if len(keys) == 1 and next(iter(keys)) in manual_stems:
        return "manual_rule"

    return "shared_root"


def rebuild_clusters(concepts: list[dict]) -> list[dict]:
    by_lemma = {c["lemma"]: c for c in concepts}
    groups: dict[str, list[str]] = defaultdict(list)
    for lem in by_lemma:
        key = stem_key(lem)
        if key:
            groups[key].append(lem)

    clusters = []
    for key, lems in sorted(groups.items()):
        rejected_spec = FALSE_FRIENDS_BY_STEM.get(key, {})
        rejected_members = [
            {"word": w, "reason": reason} for w, reason in sorted(rejected_spec.items())
            if w in by_lemma or True  # always report known false friends for this stem
        ]
        # Only keep rejected that actually appeared in Mark inventory OR were considered
        rejected_members = [
            r for r in rejected_members if r["word"] in by_lemma or r["word"] in lems or r["word"] in rejected_spec
        ]
        # If veriga is in lems, remove from members
        uniq = sorted(set(lems) - set(rejected_spec))
        # Also record if false friend was in the raw stem group
        for w, reason in rejected_spec.items():
            if w in lems or w in by_lemma:
                pass  # already in rejected_members

        if len(uniq) < 2:
            continue

        pos_mix = [by_lemma[l]["pos"] for l in uniq]
        freqs = [by_lemma[l]["frequency"] for l in uniq]
        basis = join_basis_for(uniq, pos_mix)
        note = (
            f"CLASSLA razdvaja u {len(uniq)} lema (stem≈{key}); "
            f"join_basis={basis}"
        )
        if len(set(pos_mix)) > 1:
            note += "; mešovite vrste reči"
        if rejected_members:
            note += f"; rejected: {[r['word'] for r in rejected_members]}"

        clusters.append(
            {
                "concept_cluster": key,
                "lemmas": uniq,
                "pos_mix": pos_mix,
                "frequencies": freqs,
                "join_basis": basis,
                "rejected_members": [
                    r for r in rejected_members if r["word"] in by_lemma or r["word"] in lems
                ]
                or (
                    # still show known false friend if present in inventory
                    [
                        {"word": w, "reason": reason}
                        for w, reason in rejected_spec.items()
                        if w in by_lemma
                    ]
                ),
                "note": note,
            }
        )

    clusters.sort(key=lambda c: (-len(c["lemmas"]), -sum(c["frequencies"])))
    return clusters


def possible_missed_clusters(
    concepts: list[dict],
    clusters: list[dict],
    *,
    min_cooccur: int = 6,
    max_pairs: int = 40,
) -> list[dict]:
    """Pairs that co-occur in verses but do not share a stem cluster."""
    clustered_pairs: set[frozenset[str]] = set()
    for cl in clusters:
        for a, b in combinations(cl["lemmas"], 2):
            clustered_pairs.add(frozenset((a, b)))

    # Prefer keep+borderline for signal; include dropped narrative only if high cooccur
    candidates = [
        c
        for c in concepts
        if c["status"] in ("keep", "borderline")
        or (c["status"] == "dropped" and c["freq"] >= 20)
    ]
    verse_sets = {
        c["lemma"]: {verse_key(v) for v in c["verses"]} for c in candidates
    }
    lemmas = [c["lemma"] for c in candidates]
    stem = {l: stem_key(l) for l in lemmas}

    scored: list[tuple[int, str, str]] = []
    for a, b in combinations(lemmas, 2):
        if frozenset((a, b)) in clustered_pairs:
            continue
        sa, sb = stem[a], stem[b]
        if sa and sb and sa == sb:
            continue  # same stem — should already be clustered or false-friend
        inter = verse_sets[a] & verse_sets[b]
        n = len(inter)
        if n >= min_cooccur:
            scored.append((n, a, b))

    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    status = {c["lemma"]: c["status"] for c in concepts}
    reason = {c["lemma"]: c.get("drop_reason") for c in concepts}
    out = []
    for n, a, b in scored:
        # Skip pure narrative co-occurrence (reći+ići) — low review value
        if reason.get(a) == "narrative_noise" and reason.get(b) == "narrative_noise":
            continue
        # Prefer at least one keep/borderline
        if status.get(a) == "dropped" and status.get(b) == "dropped":
            continue
        out.append(
            {
                "words": [a, b],
                "co_occurrence": n,
                "statuses": [status.get(a), status.get(b)],
                "note": (
                    "često u istim stihovima Marka, ne dele koren u klaster-logici "
                    "— proveriti (sinonim / supletiv / fraza / slučajna ko-pojava)"
                ),
            }
        )
        if len(out) >= max_pairs:
            break
    return out


def main() -> None:
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    concepts_out = []
    counts = {"keep": 0, "dropped": 0, "borderline": 0}
    drop_reason_counts: dict[str, int] = defaultdict(int)

    for c in v1["concepts"]:
        status, reason = classify_concept(c)
        counts[status] += 1
        if reason:
            drop_reason_counts[reason] += 1
        concepts_out.append(
            {
                "lemma": c["lemma"],
                "status": status,
                "drop_reason": reason,
                "freq": c["frequency"],
                "forms": c["forms"],
                "pos": c["pos"],
                "corpus": c["corpus"],
                "n_verses": c.get("n_verses", len(c["verses"])),
                "verses": [
                    {"chapter": v["chapter"], "verse": v["verse"]}
                    for v in c["verses"]
                ],
            }
        )

    # Ensure frequency field name consistency for cluster rebuild
    for c in concepts_out:
        c["frequency"] = c["freq"]

    clusters = rebuild_clusters(concepts_out)
    missed = possible_missed_clusters(concepts_out, clusters)

    # Strip helper frequency duplicate for final schema (keep freq)
    for c in concepts_out:
        c.pop("frequency", None)

    out = {
        "book": "Марко",
        "regenerated_at": datetime.now(timezone.utc).isoformat(),
        "source_v1": "data/concept/finetune_candidates_marko_v1.json",
        "note": (
            "Transparent filter pass over v1 inventory. "
            "status=keep → ručni shortlist kandidat; "
            "borderline → freq 2–4 (vidljiv rep); "
            "dropped → narrative_noise / proper_noun. "
            "Stopwords i većina PROPN nikad nisu ušli u v1 (CLASSLA pre-filter) "
            "— nisu 'tiho izbačeni' ovde, već ranije u ekstrakciji."
        ),
        "n_total": len(concepts_out),
        "n_keep": counts["keep"],
        "n_dropped": counts["dropped"],
        "n_borderline": counts["borderline"],
        "n_clusters": len(clusters),
        "n_possible_missed_clusters": len(missed),
        "filter_legend": {
            "keep": "freq≥5, nije narrative_noise, nije ne-teološki PROPN",
            "borderline": "freq 2–4 — zona odsecanja, ne za trening dok se ručno ne potvrdi",
            "dropped.narrative_noise": "čest narativni glagol/pridev/imenica (reći, doći, dan…)",
            "dropped.proper_noun": "vlastita imenica van teološkog whitelista (bog/hrist/…)",
            "pre_filter_not_in_list": (
                "stopword / većina ličnih imena i toponima — odsečeni u CLASSLA ekstrakciji v1"
            ),
        },
        "drop_reason_counts": dict(drop_reason_counts),
        "concepts": concepts_out,
        "clusters": clusters,
        "possible_missed_clusters": missed,
    }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(
        f"n_total={out['n_total']} keep={out['n_keep']} "
        f"dropped={out['n_dropped']} borderline={out['n_borderline']}"
    )
    print(f"clusters={out['n_clusters']} possible_missed={out['n_possible_missed_clusters']}")
    print("drop_reason_counts:", dict(drop_reason_counts))
    print("\nSample dropped narrative_noise:")
    for c in concepts_out:
        if c["drop_reason"] == "narrative_noise":
            print(f"  {c['freq']:3d} {c['lemma']}")
            break
    print("Sample borderline:")
    for c in concepts_out:
        if c["status"] == "borderline":
            print(f"  {c['freq']:3d} {c['lemma']} ({c['pos']})")
            break
    print("\nCluster with rejected_members:")
    for cl in clusters:
        if cl["rejected_members"]:
            print(f"  {cl['concept_cluster']}: {cl['lemmas']} rejected={cl['rejected_members']}")
    print("\nTop possible_missed:")
    for m in missed[:10]:
        print(f"  {m['co_occurrence']:3d} {m['words']}")


if __name__ == "__main__":
    main()
