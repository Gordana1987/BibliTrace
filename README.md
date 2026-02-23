# BibliTrace

**Repository:** [github.com/Gordana1987/BibliTrace](https://github.com/Gordana1987/BibliTrace)

Web tool that detects Biblical intertextuality in Serbian literary texts (Daničić–Karadžić Bible as reference).

## Setup

### Backend (Conda + Python)

```bash
cd backend
conda env create -f environment.yml
conda activate bibli_trace
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

## Stack

- **Backend:** Python, FastAPI, Uvicorn, pandas, classla, sentence-transformers
- **Frontend:** Next.js
