# BibliTrace

**Repository:** [github.com/Gordana1987/BibliTrace](https://github.com/Gordana1987/BibliTrace)

Web tool that detects Biblical intertextuality in Serbian literary texts (Daničić–Karadžić Bible as reference).

## Setup

### Backend (Conda + Python)

```bash
cd backend
conda env create -f environment.yml
conda activate bibli_trace
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

Or with pip only (from a conda env):

```bash
cd backend
conda create -n bibli_trace python=3.11
conda activate bibli_trace
pip install -r requirements.txt
python -m uvicorn app:app --reload
```

> **Note:** Use `python -m uvicorn` so the correct env’s Python is used; plain `uvicorn` can give "command not found" if it’s not on your PATH.

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

## Data pipeline (Bible corpus)

To build the Bible corpus and search index from scratch:

1. **Scrape** – Fetch OT+NT from Svetosavlje.org (Cyrillic):
   ```bash
   python scripts/scrape_bible.py
   ```
   Output: `backend/data/bible/bible.csv`

2. **Lemmatize** – Run CLASSLA on verse text (~20 min, first run downloads model):
   ```bash
   python scripts/lemmatize_bible.py
   ```
   Output: `backend/data/bible/bible_lemmatized.csv`

3. **Build BM25 index** – For lexical retrieval:
   ```bash
   python scripts/build_bm25_index.py
   ```
   Output: `backend/data/bible/bm25_index.joblib`

4. **Build embedding indexes** – For semantic search (Qwen3 default; LaBSE for comparison):
   ```bash
   python scripts/build_embeddings.py qwen
   python scripts/build_embeddings.py labse
   ```
   Or `python scripts/build_embeddings.py both`. First run downloads models (~1.2GB Qwen, ~470MB LaBSE).

CSV and index files are gitignored; run the pipeline locally. Query input must be **Cyrillic** (Latin is not supported).

## Retrieval pipeline (current behavior)

- **Hybrid search:**  
  - BM25 over the lemmatized corpus returns up to 200 candidate verses.  
  - Qwen3-Embedding-0.6B reranks those candidates semantically and returns the top 20.  
  - Optionally, LaBSE reranks the *same* BM25 candidates for side‑by‑side comparison.

- **BM25 query handling:**  
  - Queries are lemmatized with CLASSLA and tokenized.  
  - We also tokenize the **raw query text** and search with the **union** of lemma + raw tokens to reduce lemma mismatches (e.g. „синови грома“ vs „син гром“).

- **Phrase boosting:**  
  - If the exact query phrase appears in a verse (e.g. „синови грома“ in Mk 3:17), that verse is always included in the candidate set and phrase matches are ranked first in the final results (ordered among themselves by semantic score).

- **Data cleaning at retrieval time:**  
  - Editorial headlines and duplicate verse references are skipped.  
  - Pagination artefacts like `Pages:` / page numbers are ignored.  
  - Liturgical markers `*` and `†` are stripped from verse text before returning matches.

## Known future improvements (ideas)

- **Genealogical / list verse downweighting:**  
  Query‑aware reduction of noisy genealogy and census lists (1 Дн, 2 Језд, etc.) so they don’t dominate BM25, while still allowing direct searches for list/census content.

- **Phrase vs semantic mix tuning:**  
  Empirically tune how many phrase matches to surface vs. pure semantic neighbors, and possibly label them differently in the UI.

- **Stricter normalization consistency:**  
  Unify the normalization pipeline used for building `bible_lemmatized.csv`, BM25, and live queries (beyond the current lemma+raw token workaround).

## Stack

- **Backend:** Python, FastAPI, Uvicorn, pandas, classla, rank_bm25, sentence-transformers (Qwen3, LaBSE)
- **Frontend:** Next.js
