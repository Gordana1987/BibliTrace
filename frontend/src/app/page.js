"use client";

import { useState } from "react";
import { searchConcept } from "@/lib/api";
import TextInput from "@/components/TextInput";
import ResultsPanel from "@/components/ResultsPanel";
import { NT_ALL_BOOKS } from "@/lib/ntBooks";

const PAGE_SIZE = 20;

export default function Home() {
  const [term, setTerm] = useState("");
  const [mode, setMode] = useState("exact");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMoreByCorpus, setLoadingMoreByCorpus] = useState({});
  const [corpora, setCorpora] = useState(["dk", "spc"]);
  const [books, setBooks] = useState(() => [...NT_ALL_BOOKS]);

  function booksPayload(selected) {
    if (!selected?.length || selected.length === NT_ALL_BOOKS.length) {
      return undefined;
    }
    return selected;
  }

  async function runSearch(nextMode = mode, nextCorpora = corpora, nextBooks = books) {
    const q = term.trim();
    if (!q || nextCorpora.length === 0 || nextBooks.length === 0) return;
    setError(null);
    setResult(null);
    setLoadingMoreByCorpus({});
    setLoading(true);
    try {
      const data = await searchConcept({
        term: q,
        mode: nextMode,
        corpora: nextCorpora,
        books: booksPayload(nextBooks),
        offset: 0,
        limit: PAGE_SIZE,
      });
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleSearch() {
    await runSearch(mode, corpora, books);
  }

  async function handleTrySemantic() {
    setMode("semantic");
    await runSearch("semantic", corpora, books);
  }

  async function handleLoadMore(corpus) {
    if (!term.trim() || !result || !corpora.includes(corpus) || books.length === 0) {
      return;
    }
    const current = result.results_by_corpus?.[corpus];
    if (!current) return;
    const offset = current.hits?.length || 0;
    if (offset >= (current.total || 0)) return;

    setLoadingMoreByCorpus((prev) => ({ ...prev, [corpus]: true }));
    setError(null);
    try {
      const data = await searchConcept({
        term: term.trim(),
        mode: result.mode,
        corpora: [corpus],
        books: booksPayload(books),
        offset,
        limit: PAGE_SIZE,
      });
      const page = data.results_by_corpus?.[corpus];
      if (!page) return;
      setResult((prev) => {
        if (!prev) return data;
        const prevCorpus = prev.results_by_corpus?.[corpus] || current;
        return {
          ...prev,
          results_by_corpus: {
            ...prev.results_by_corpus,
            [corpus]: {
              ...prevCorpus,
              ...page,
              hits: [...(prevCorpus.hits || []), ...(page.hits || [])],
              returned: (prevCorpus.hits?.length || 0) + (page.hits?.length || 0),
              offset: 0,
            },
          },
        };
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoadingMoreByCorpus((prev) => ({ ...prev, [corpus]: false }));
    }
  }

  return (
    <main className="page">
      <header className="header">
        <h1>Видело</h1>
        <p>Претрага појмова и тема у Новом завету</p>
      </header>
      <TextInput
        value={term}
        onChange={setTerm}
        mode={mode}
        onModeChange={setMode}
        onSearch={handleSearch}
        disabled={loading}
        corpora={corpora}
        onCorporaChange={setCorpora}
        books={books}
        onBooksChange={setBooks}
      />
      <ResultsPanel
        result={result}
        error={error}
        loading={loading}
        loadingMoreByCorpus={loadingMoreByCorpus}
        selectedCorpora={corpora}
        onLoadMore={handleLoadMore}
        onTrySemantic={handleTrySemantic}
      />
    </main>
  );
}
