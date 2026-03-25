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

## Known model weaknesses

These are systematic failure patterns observed during testing, not bugs — they reflect fundamental limitations of the current embedding models.

- **Syntactic pattern dominance (LaBSE):**
  LaBSE tends to latch onto syntactic surface patterns rather than content. Queries with strong syntactic markers (e.g. negation `"ни... ни..."`, conditional `"ако..."`, vocative `"Горе..."`) cause LaBSE to retrieve verses that share the pattern but are semantically unrelated. Qwen3 is less susceptible to this. Mitigation: fine-tuning pairs that teach the model to prioritize content over syntax.

- **Privative / negation constructions:**
  Queries like `"љубави туђ"` (alien to love) retrieve love-related verses instead of hatred/fratricide verses. Embedding models average meaning — the dominant noun (`љубав`) overrides the privative construction. Affects any genitive-of-privation or `"без X"` / `"лишен X"` pattern common in medieval Serbian literature. Mitigation: separate fine-tuning pairs category for privative constructions.

- **Single-word substitution paraphrases:**
  When a literary author replaces one word in a Bible quote (e.g. `"пролеће"` instead of `"зиму"` in Пс 74:17), neither BM25 nor semantic models reliably find the source verse. BM25 misses because the substituted word doesn't match; semantic models miss because the embedding shift is too subtle. Mitigation: fine-tuning pairs; knowledge layer for known liturgical variants.

- **Short query ranking precision:**
  With 2–3 token queries (e.g. `"синови грома"`), BM25 scores are flat across many candidates and genealogy/census verses dominate due to high term frequency of common words. Semantic reranking helps but doesn't fully compensate. Mitigation: `k1` tuning; genealogical list downweighting.

- **Narrative context blindness:**
  Models don't know the story behind a name or event. `"Каин, љубави туђ"` implies fratricide, jealousy, and the theological contrast with love — but models see surface tokens. Verses thematically central to the Cain story but without shared vocabulary are systematically missed (e.g. 1 Јован 3:12 from DK). Mitigation: knowledge graph; fine-tuning pairs encoding narrative connections.

## Known future improvements (ideas)

- **Genealogical / list verse downweighting:**  
  Query‑aware reduction of noisy genealogy and census lists (1 Дн, 2 Језд, etc.) so they don’t dominate BM25, while still allowing direct searches for list/census content.

- **Phrase vs semantic mix tuning:**  
  Empirically tune how many phrase matches to surface vs. pure semantic neighbors, and possibly label them differently in the UI.

- **Stricter normalization consistency:**  
  Unify the normalization pipeline used for building `bible_lemmatized.csv`, BM25, and live queries (beyond the current lemma+raw token workaround).

- **Cross-corpus score normalization (`version=both`):**  
  When searching both DK and Bakotić simultaneously, each corpus currently normalizes scores against its own maximum (so both have a top score ≈ 1.0). This makes cross-corpus ranking approximate.  
  Fix: return raw cosine similarity scores from `_run_semantic_rerank` (no per-corpus scaling), then normalize once across the merged pool. Raw scores are directly comparable because both corpora use the same embedding model in the same vector space.

- **BM25 `k1` tuning:**  
  Lower `k1` from default 1.5 to ~0.3–0.5 to reduce term-frequency dominance. Bible verses are short (1–2 sentences), so a token appearing twice in a verse is mostly noise rather than a relevance signal. Requires rebuilding BM25 indexes.

- **Versification mapper:**  
  Lookup table to align verse numbers between DK and Bakotić (and future corpora) for cross-corpus comparison. Needed for cases like LXX additions (present in DK, absent in Bakotić) and minor prophet versification differences.

- **Book name mapper:**  
  Canonical book ID (e.g. `GEN`, `MK`) mapped from each corpus's book names, to enable cross-corpus search by book name and future side-by-side comparison UI.

- **Fine-tuning embedding models on labeled pairs:**
  Fine-tune Qwen3 and/or LaBSE on labeled query→verse pairs collected during testing. Including both DK and Bakotić versions of the same verse as co-positives teaches the model dialect equivalence (`дажд` ≈ `киша`, `љето` ≈ `лето`, `Исус Навин` ≈ `Исус` in Joshua context) — something the base models don't know. Requires ~500–1000 quality labeled pairs minimum. Pairs should be collected in two categories: direct allusion pairs and privative/negation pairs (see model weaknesses). Start collecting during testing; fine-tuning becomes worthwhile once 200–300 confirmed pairs are available.

- **Cross-corpus deduplication (`version=both`):**  
  When the same canonical verse appears from both DK and Bakotić, collapse into a single result showing both translations side by side. Requires the book name mapper to identify matches reliably. Short-term workaround: deduplicate by `(chapter, verse)` with fuzzy book name matching, keeping the higher-scoring result.

## Planned corpora

| Corpus | Status | Notes |
|--------|--------|-------|
| Daničić–Karadžić (DK) | ✓ Active | Full OT+NT, Ijekavian, Masoretic/Protestant tradition |
| Bakotić | ✓ Active | Full OT+NT, Ekavian, Protestant tradition |
| SPC Sinod NT | Planned | NT only, modern Serbian Orthodox liturgical translation; scrapable from rastko.rs |
| Atanasije Psalter | Planned | 150 Psalms, Orthodox/Septuagint tradition; requires digitizing from physical book |

## Known data issues (fix in next pipeline rebuild)

1. **Bakotić Psalms — wrong chapter numbers**
   All 150 Psalms scraped as `chapter=1`. Fix: add `PSALM_RE = re.compile(r"^\s*Псалам\s+(\d+)\.?\s*$")` to `scrape_bible_bakotic.py` so each Psalm number becomes the chapter.

2. **DK editorial headlines with verse numbers**
   Some section headings in DK were assigned verse numbers during scraping (e.g. `"Јотор походи Мојсија. Постављање судија."` stored as 2 Мој 17:18). The duplicate-reference filter only catches headlines without a unique reference. Fix: identify and remove these during scraping or add a text-based headline detector (all-caps, ends with period but no verb, etc.).

## Known test cases

### `"Лето и пролеће Господ сазда"` → Псалм 74:17
- **Status:** currently fails in DK and Bakotić
- **DK** (Пс 74:17): `"Ти си утврдио све крајеве земаљске, љето и зиму ти си уредио."` — different season (зима) and Ijekavian (љето)
- **Bakotić** (Пс 74:17): `"Ти си утврдио све крајеве земље; лето и зиму ти си уредио."` — different season (зима)
- **Atanasije Psalter** (Пс 74:17): `"лето и пролеће"` — exact match
- **Explanation:** Лазаревић quotes from the Orthodox/Septuagint liturgical tradition. The Atanasije Psalter uses `"лето и пролеће"` which matches the literary phrase directly. This test case will only work correctly once the Psalter is added.

## Stack

- **Backend:** Python, FastAPI, Uvicorn, pandas, classla, rank_bm25, sentence-transformers (Qwen3, LaBSE)
- **Frontend:** Next.js
