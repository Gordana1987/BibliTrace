"use client";

import { useState } from "react";
import { searchConcept } from "@/lib/api";
import TextInput from "@/components/TextInput";
import ResultsPanel from "@/components/ResultsPanel";

const PAGE_SIZE = 20;

export default function Home() {
  const [term, setTerm] = useState("");
  const [mode, setMode] = useState("exact");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMoreByCorpus, setLoadingMoreByCorpus] = useState({});
  const [corpora, setCorpora] = useState(["dk", "spc"]);

  async function runSearch(nextMode = mode, nextCorpora = corpora) {
    const q = term.trim();
    if (!q || nextCorpora.length === 0) return;
    setError(null);
    setResult(null);
    setLoadingMoreByCorpus({});
    setLoading(true);
    try {
      const data = await searchConcept({
        term: q,
        mode: nextMode,
        corpora: nextCorpora,
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
    await runSearch(mode, corpora);
  }

  async function handleTrySemantic() {
    setMode("semantic");
    await runSearch("semantic", corpora);
  }

  async function handleLoadMore(corpus) {
    if (!term.trim() || !result || !corpora.includes(corpus)) return;
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
        <h1>BibliTrace</h1>
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
