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

CSV and index files are gitignored; run the pipeline locally. Query input must be **Cyrillic** (Latin is not supported).

## Stack

- **Backend:** Python, FastAPI, Uvicorn, pandas, classla, rank_bm25, scikit-learn
- **Frontend:** Next.js
