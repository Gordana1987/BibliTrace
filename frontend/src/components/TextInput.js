"use client";

import { ACTIVE_CORPORA, CORPUS_LABELS } from "../lib/corpora";

export default function TextInput({
  value,
  onChange,
  onAnalyze,
  disabled,
  corpora,
  onCorporaChange,
}) {
  const canAnalyze = value.trim() && corpora.length > 0;

  function toggleCorpus(id) {
    const next = corpora.includes(id)
      ? corpora.filter((c) => c !== id)
      : [...corpora, id];
    if (next.length > 0) {
      onCorporaChange(next);
    }
  }

  return (
    <section className="input-section">
      <label htmlFor="text" className="input-label">
        Текст за анализу (ћирилица)
      </label>
      <textarea
        id="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Налепите или унесите текст…"
        rows={10}
        disabled={disabled}
      />
      <div className="input-controls">
        <div className="corpus-chips-wrap">
          <span className="corpus-chips-label">Корпуси</span>
          <div className="corpus-chips" role="group" aria-label="Избор корпуса">
            {ACTIVE_CORPORA.map((id) => {
              const active = corpora.includes(id);
              return (
                <button
                  key={id}
                  type="button"
                  className={`corpus-chip${active ? " corpus-chip--active" : ""}`}
                  disabled={disabled}
                  aria-pressed={active}
                  onClick={() => toggleCorpus(id)}
                >
                  {CORPUS_LABELS[id]}
                </button>
              );
            })}
          </div>
        </div>
        <button
          type="button"
          className="analyze-btn"
          onClick={onAnalyze}
          disabled={disabled || !canAnalyze}
        >
          Analyze
        </button>
      </div>
    </section>
  );
}
