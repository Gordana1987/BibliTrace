# BibliTrace

**Repository:** [github.com/Gordana1987/BibliTrace](https://github.com/Gordana1987/BibliTrace)

Web tool that detects Biblical intertextuality in Serbian literary texts. The **DK reference corpus** is built from **JW.org sr-latn** (Daničić–Karadžić), transliterated to Cyrillic locally — see [Corpus sources](#corpus-sources-decision). You can search **one or more** corpora (DK, Bakotić, SPC) from the UI.

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

Open **http://localhost:3000** (API default: **http://127.0.0.1:8000**). Queries must be **Cyrillic**.

## Data pipeline (Bible corpus)

Corpus CSVs (`bible.csv`, `bible_lemmatized.csv` per corpus) are in the repo. BM25/embedding indexes and other artefacts are gitignored — build them locally after `conda activate bibli_trace` from `backend/`.

### DK (`data/bible/`)

1. **Scrape JW Latin → Cyrillic CSV** (~2 h with `--delay 2`, resume on failure):
   ```bash
   python scripts/scrape_bible_jw_latn.py --delay 2.0
   # resume: python scripts/scrape_bible_jw_latn.py --resume --delay 2.0
   ```
   Source: [JW sr-latn Daničić–Karadžić](https://www.jw.org/sr-latn/biblioteka/sveto-pismo/dani%C4%8Di%C4%87-karad%C5%BEi%C4%87/knjige/). Transliteration: `latin_to_cyrillic.py` (not JW’s sr-cyrl). Optional `lj`/`nj` review: `extract_lj_nj_review.py`.

   Output: `backend/data/bible/bible.csv` (~31k verses, 66 books).

   Legacy Svetosavlje scraper (deprecated for DK): `scrape_bible.py`, cleanup helper `clean_bible_csv.py`.

2. **Lemmatize** – CLASSLA (~20 min, first run downloads model):
   ```bash
   python scripts/lemmatize_bible.py
   ```
   Output: `backend/data/bible/bible_lemmatized.csv`

3. **Build BM25 index:**
   ```bash
   python scripts/build_bm25_index.py
   ```

4. **Build embedding indexes** (Qwen3 default; LaBSE optional in UI):
   ```bash
   python scripts/build_embeddings.py both
   ```

### Bakotić (`data/bakotic/`)

```bash
python scripts/scrape_bible_bakotic.py
python scripts/lemmatize_bible.py --corpus bakotic
python scripts/build_bm25_index.py --corpus bakotic
python scripts/build_embeddings.py both --corpus bakotic
```

### SPC (`data/spc/`)

```bash
python scripts/ingest_spc_epub.py --download
python scripts/lemmatize_bible.py --corpus spc
python scripts/build_bm25_index.py --corpus spc
python scripts/build_embeddings.py both --corpus spc
```

### API corpus selection

`POST /api/analyze` accepts `corpora`: any non-empty subset of `["dk", "bakotic", "spc"]` (e.g. `["dk"]`, `["bakotic", "spc"]`, all three). Legacy field `version` (`both`, `all`) still works for older clients.

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

- **Cross-corpus score normalization (multi-corpus search):**  
  Scores are normalized once across the merged result pool after reranking (same embedding model for all corpora).

- **BM25 `k1` tuning:**  
  Lower `k1` from default 1.5 to ~0.3–0.5 to reduce term-frequency dominance. Bible verses are short (1–2 sentences), so a token appearing twice in a verse is mostly noise rather than a relevance signal. Requires rebuilding BM25 indexes.

- **Versification mapper:**  
  Lookup table to align verse numbers between DK and Bakotić (and future corpora) for cross-corpus comparison. Needed for cases like LXX additions (present in DK, absent in Bakotić) and minor prophet versification differences.

- **Book name mapper:**  
  Canonical book ID (e.g. `GEN`, `MK`) mapped from each corpus's book names, to enable cross-corpus search by book name and future side-by-side comparison UI.

- **Fine-tuning embedding models on labeled pairs:**
  Fine-tune Qwen3 and/or LaBSE on labeled query→verse pairs collected during testing. Including both DK and Bakotić versions of the same verse as co-positives teaches the model dialect equivalence (`дажд` ≈ `киша`, `љето` ≈ `лето`, `Исус Навин` ≈ `Исус` in Joshua context) — something the base models don't know. Requires ~500–1000 quality labeled pairs minimum. Pairs should be collected in two categories: direct allusion pairs and privative/negation pairs (see model weaknesses). Start collecting during testing; fine-tuning becomes worthwhile once 200–300 confirmed pairs are available.

- **Cross-corpus deduplication:**  
  When the same canonical verse appears from multiple corpora, collapse into a single result showing translations side by side. Requires the book name mapper.

## Corpus sources (decision)

Single place for **what we ingest** and **what we reject**. This supersedes ad-hoc notes elsewhere.

### Corpus 1 — Daničić–Karadžić (DK) — **in use**

| Item | Detail |
|------|--------|
| **Current ingest** | **JW.org `sr-latn`** — one HTML page per chapter; `scrape_bible_jw_latn.py` → `latin_to_cyrillic.py` → `data/bible/bible.csv`. |
| **Why Latin** | Good ijekavian Daničić–Karadžić text; we transliterate ourselves (not JW `sr-cyrl`). |
| **Legal** | Watch Tower / JW site ToS — use for **personal research**; do not redistribute scraped CSV publicly without checking rights. |
| **Alternate (not wired)** | Rastko Daničić OT ([collection 10094](https://www.rastko.rs/bogoslovlje/delo/10094), 39 `delo` IDs, skip `10477`) + Wikisource Karadžić 1847 NT — harder HTML parsing than JW; kept as fallback plan. |

### Corpus 2 — Bakotić

Done (Wikisource); see `scrape_bible_bakotic.py` and `backend/data/bakotic/`.

### Corpus 3 — SPC full Bible — **in use**

| Item | Detail |
|------|--------|
| **Preferred ingest** | **EPUB** from [Источник / Епархија канадска](https://istocnik.ca/sveto-pismo) — official free file (2014 SPC printed edition). Build CSV with `backend/scripts/ingest_spc_epub.py` (`--download` fetches into `data/spc/source.epub`, then parses to `data/spc/bible.csv`). |
| **Alternate / legacy** | **svetosavlje.org** — [sveto-pismo](https://svetosavlje.org/sveto-pismo/) only (valid online source for this edition); messy HTML if you scrape instead of EPUB. |
| **Scraper (HTML path)** | Section headings are **inconsistently** marked in HTML — detect verse lines by **verse-number pattern** (digit + period/dot at line start), **not** by assuming a specific HTML tag for headings vs body. |
| **Canon** | Full Orthodox canon including **11 deuterocanonical** books: Товит, Јудита, Премудрости Соломонове, Сирах, Варух, Посланица Јеремијина, 1–4 Макавејске, 2 Јездрина. |
| **Attribution** | OT = modernized Daničić; deuterocanonicals = Митрополит Амфилохије Радовић + Епископ Атанасије Јевтић (1995); NT = Комисија САС (1984). |

**Do not use** `pravoslavna-srbija.com` for SPC — unofficial **ekavian** adaptation of unclear provenance.

### Explicitly rejected sources

| Source | Reason |
|--------|--------|
| **svetosavlje.org** for **DK** | Modernized spelling; messy HTML (headlines vs verses). |
| **pravoslavna-srbija.com** (SPC) | Unofficial ekavian adaptation. |
| **biblija.rs** PDF (Глас мира 2012) | Syllabic layout, inline cross-refs — not cleanly parseable as verse CSV. |
| **eBible.org** `srp1868` etc. | Not used for DK (project choice). |
| **Rastko.rs** for **NT** | Encoding issues — unusable for NT. |
| **JW.org** (redistribution) | Scraped DK for local index only; not a substitute for public-domain redistribution. |

## Planned corpora

| Corpus | Status | Notes |
|--------|--------|-------|
| Daničić–Karadžić (DK) | ✓ Active | JW sr-latn → Cyrillic; `scrape_bible_jw_latn.py` |
| Bakotić | ✓ Active | Wikisource Ekavian; `scrape_bible_bakotic.py` |
| SPC | ✓ Active | Istocnik EPUB; `ingest_spc_epub.py` |
| Atanasije Psalter | Planned | 150 Psalms; physical / digitization |

## Known data issues (fix in next pipeline rebuild)

1. **Bakotić Psalms — wrong chapter numbers**
   All 150 Psalms scraped as `chapter=1`. Fix: `PSALM_RE` in `scrape_bible_bakotic.py` (Псалам *n* as chapter).

2. **Legacy Svetosavlje DK** — replaced by JW DK pipeline; `scrape_bible.py` kept for reference only.

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
