"use client";

import { useState } from "react";
import { analyzeText } from "@/lib/api";
import TextInput from "@/components/TextInput";
import ResultsPanel from "@/components/ResultsPanel";

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingLabseByCorpus, setLoadingLabseByCorpus] = useState({});
  const [corpora, setCorpora] = useState(["dk", "spc"]);

  async function handleAnalyze() {
    setError(null);
    setResult(null);
    setLoadingLabseByCorpus({});
    setLoading(true);
    try {
      const data = await analyzeText(text, false, corpora);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleLoadLabse(corpus) {
    if (!text.trim() || !result || !corpora.includes(corpus)) return;
    setLoadingLabseByCorpus((prev) => ({ ...prev, [corpus]: true }));
    setError(null);
    try {
      const data = await analyzeText(text, true, [corpus]);
      setResult((prev) => {
        if (!prev) return data;
        const nextLabse = {
          ...(prev.labse_matches_by_corpus || {}),
          [corpus]:
            data.labse_matches_by_corpus?.[corpus] || data.labse_matches || [],
        };
        return {
          ...prev,
          labse_matches_by_corpus: nextLabse,
          labse_matches:
            corpora.length === 1 ? nextLabse[corpus] : prev.labse_matches,
        };
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoadingLabseByCorpus((prev) => ({ ...prev, [corpus]: false }));
    }
  }

  return (
    <main className="page">
      <header className="header">
        <h1>BibliTrace</h1>
        <p>Откривање библијске интертекстуалности у српским књижевним текстовима</p>
      </header>
      <TextInput
        value={text}
        onChange={setText}
        onAnalyze={handleAnalyze}
        disabled={loading}
        corpora={corpora}
        onCorporaChange={setCorpora}
      />
      <ResultsPanel
        result={result}
        error={error}
        loading={loading}
        loadingLabseByCorpus={loadingLabseByCorpus}
        selectedCorpora={corpora}
        onLoadLabse={handleLoadLabse}
      />
    </main>
  );
}
