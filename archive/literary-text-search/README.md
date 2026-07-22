# Archive: literary-text search (pre–concept-search)

This folder preserves the **previous product direction**: detect biblical intertextuality in **literary passages** (quotes, motifs, paraphrases), not concept/term search.

Live BibliTrace (2026-07 onward) is **concept-search** over the New Testament (`exact` / `lemma` / `semantic`). Do not treat this archive as the live API or UI.

## What lives here

| Path | Contents | Git |
|------|----------|-----|
| `benchmark/golden_set.json` | Adversarial literary golden set (~37 cases) | tracked |
| `benchmark/golden_random.json` | Random/incidental control set | tracked |
| `benchmark/_cases_*.json` | Source fragments for `materialize_golden_sets.py` | tracked |
| `benchmark/hyde_dense_cases.json` | HyDE paraphrase cases | tracked |
| `benchmark/results/` | Timestamped eval/diag JSON + logs (regenerable) | **gitignored** |
| `scripts/` | Golden eval + Phase A/BM25 + encode/expansion/CE/HyDE diags | tracked |

Active corpora indexes (`backend/data/dk/`, `backend/data/spc/`) were **not** moved. Live detection code remains under `backend/services/detection.py` until the concept-search refactor replaces the analyze path.

## How to re-run (historical)

From repo root, with the backend venv active:

```bash
cd backend
source .venv/bin/activate   # or conda activate bibli_trace

# Baseline over archived golden (writes into archive/.../benchmark/results/)
python ../archive/literary-text-search/scripts/run_golden_set.py

# BM25-only Phase A
python ../archive/literary-text-search/scripts/run_phase_a_bm25.py

# Other diags (encode, expansion, CE, dense, HyDE) — same pattern
python ../archive/literary-text-search/scripts/run_cross_encoder_diag.py
```

Scripts add `backend/` to `sys.path` and read golden files from this archive tree. They still call the **current** `services.detection` / config — if live search diverges far enough, archived evals may need a frozen detection snapshot.

## Why archived

- Product pivot: **pojmovna pretraga** (term + mode), not literary-text echo detection.
- Golden sets and diags assume long literary queries, OT/NT mix in cases, phrase-suppression probes, etc. — not concept modes.
- Keeps history and regenerable methodology without cluttering the live tree.
