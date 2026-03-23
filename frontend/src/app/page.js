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
  const [loadingLabse, setLoadingLabse] = useState(false);
  const [version, setVersion] = useState("dk");

  async function handleAnalyze() {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const data = await analyzeText(text, false, version);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleCompareWithLabse() {
    if (!text.trim() || !result) return;
    setLoadingLabse(true);
    setError(null);
    try {
      const data = await analyzeText(text, true, version);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoadingLabse(false);
    }
  }

  return (
    <main className="page">
      <header className="header">
        <h1>BibliTrace</h1>
        <p>Detect Biblical intertextuality in Serbian literary texts (Cyrillic)</p>
      </header>
      <TextInput
        value={text}
        onChange={setText}
        onAnalyze={handleAnalyze}
        disabled={loading}
        version={version}
        onVersionChange={setVersion}
      />
      <ResultsPanel
        result={result}
        error={error}
        loading={loading}
        loadingLabse={loadingLabse}
        onCompareWithLabse={handleCompareWithLabse}
      />
    </main>
  );
}
