"""One-shot helper: merge case arrays with layers/wrappers and write golden JSON files."""

from __future__ import annotations

import json
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "data" / "benchmark"

LAYER_BY_ID: dict[str, str] = {
    "rastko-danseti-postanje1": "literal",
    "rastko-danseti-prah": "motif",
    "desanka-pomilovanje": "literal",
    "narodne-nemanjici-pokajanje": "literal",
    "narodne-nemanjici-krst": "literal",
    "jefimija-zmija-venac": "motif",
    "slovo-ljubve-gelvuj": "literal",
    "slovo-ljubve-vode": "motif",
    "lalic-kanoni-ratnik": "literal",
    "lalic-kanoni-more": "motif",
    "lalic-kanoni-ana": "literal",
    "lalic-kanoni-ogradjeni-vrt": "motif",
    "lalic-kanoni-prazan-grob": "motif",
    "lalic-kanoni-ruka-isaija": "literal",
    "lalic-kanoni-mrtvaci-isaija": "literal",
    "lalic-kanoni-jona": "motif",
    "lalic-kanoni-jona-izbljuje": "literal",
    "lalic-kanoni-zena-patmos": "motif",
    "lalic-kanoni-danilo-tri-mladica": "literal",
    # phrase-suppression / composite
    "njegos-luca-mikrokozma-iskra": "motif",
    "njegos-gorski-vijenac-getsemanija": "motif",
    "andric-na-drini-cuprija-potop": "motif",
    "narodna-knezeva-vecera-tajna-vecera": "motif",
    "kostic-santa-maria-strasni-sud": "motif",
    "nastasijevic-sedam-krugova-andjeo-truba": "motif",
    "jefimija-ugljesa-obitelji-grob": "motif",
    "miljkovic-rec-rodjena-iz-tame": "motif",
    # verse-pinning
    "kostic-samson-i-delila-sesta-noc": "literal",
    "kostic-samson-i-delila-astarota-stub": "literal",
    # single-anchor splits
    "slovo-ljubve-vode-split-a": "motif",
    "slovo-ljubve-vode-split-b": "motif",
    "knezeva-vecera-split-a": "motif",
    "knezeva-vecera-split-b": "motif",
    "santa-maria-split-a": "motif",
    "santa-maria-split-b": "motif",
    "jefimija-ugljesa-split-a": "motif",
    "jefimija-ugljesa-split-b": "motif",
}

# Corpus book-name fixes for random set (Pouke naming)
RANDOM_BOOK_FIX = {
    "Дела": "Дела апостолска",
    "1. Царевима": "1. Краљевима",
}


def _with_layers(cases: list[dict]) -> list[dict]:
    out = []
    for case in cases:
        c = dict(case)
        c["layer"] = LAYER_BY_ID.get(c["id"], "motif")
        out.append(c)
    return out


def _fix_random_books(cases: list[dict]) -> list[dict]:
    out = []
    for case in cases:
        c = json.loads(json.dumps(case, ensure_ascii=False))
        for exp in c.get("expected", []):
            book = exp.get("book", "")
            if book in RANDOM_BOOK_FIX:
                exp["book"] = RANDOM_BOOK_FIX[book]
        c["layer"] = "incidental"
        out.append(c)
    return out


def main() -> None:
    main_cases = json.loads((BENCH / "_cases_main.json").read_text(encoding="utf-8"))
    random_cases = json.loads((BENCH / "_cases_random.json").read_text(encoding="utf-8"))

    golden_main = {
        "version": 3,
        "description": "Adversarial golden set for BibliTrace retrieval baseline. Book names match data/dk/bible.csv.",
        "benchmark": {
            "mode": "baseline",
            "set_id": "main",
            "skip_query_prefix": "TODO",
            "corpus_passes": [["dk_ekav"], ["dk"], ["dk", "dk_ekav"]],
            "notes": (
                "Adversarially curated cases incl. phrase-suppression, verse-pinning, "
                "and single-anchor-split probes. Optional diagnosis documents known miss "
                "patterns (does not affect scoring). Do not merge with golden_random.json."
            ),
        },
        "cases": _with_layers(main_cases),
    }

    golden_random = {
        "version": 1,
        "description": "Control sample: incidental church-calendar paraphrases (not adversarially curated).",
        "benchmark": {
            "mode": "baseline-random",
            "set_id": "random",
            "skip_query_prefix": "TODO",
            "corpus_passes": [["dk_ekav"], ["dk"], ["dk", "dk_ekav"]],
            "notes": (
                "Random/incidental biblical paraphrases from feast-day portal texts. "
                "Separate from main golden set — compare hit rates, not in one aggregate table."
            ),
        },
        "cases": _fix_random_books(random_cases),
    }

    (BENCH / "golden_set.json").write_text(
        json.dumps(golden_main, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (BENCH / "golden_random.json").write_text(
        json.dumps(golden_random, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote golden_set.json ({len(golden_main['cases'])} cases)")
    print(f"Wrote golden_random.json ({len(golden_random['cases'])} cases)")


if __name__ == "__main__":
    main()
