# Archive: Qwen3 NZ concept-search indexes

Pre–Embedić semantic indexes for concept search (`qwen_nt_embeddings.joblib`).

Live semantic (2026-07, confirmed by `semantic_baseline_v5_embedic_large_prod.json`) uses
**`djovak/embedic-large`** and `backend/data/{dk,spc}/embedic_large_nt_embeddings.joblib`.

## Layout

| Path | Contents | Git |
|------|----------|-----|
| `dk/qwen_nt_embeddings.joblib` | DK NZ verses, Qwen3-Embedding-0.6B | **gitignored** |
| `spc/qwen_nt_embeddings.joblib` | SPC NZ verses, same model | **gitignored** |

Full-bible `qwen_embeddings.joblib` under `backend/data/` stays in place for the legacy literary `/api/analyze` path.

## Rebuild (historical)

```bash
cd backend
source .venv/bin/activate
python scripts/build_embeddings.py qwen --all --nt-only
```

Writes into this archive folder. For live indexes use:

```bash
python scripts/build_embedic_nt_embeddings.py --model large --all
```
