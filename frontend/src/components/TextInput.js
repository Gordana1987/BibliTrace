"use client";

import {
  ACTIVE_CORPORA,
  CORPUS_LABELS,
  SEARCH_MODES,
  modePlaceholder,
} from "../lib/corpora";

export default function TextInput({
  value,
  onChange,
  mode,
  onModeChange,
  onSearch,
  disabled,
  corpora,
  onCorporaChange,
}) {
  const canSearch = value.trim() && corpora.length > 0;
  const activeMode = SEARCH_MODES.find((m) => m.id === mode) || SEARCH_MODES[0];

  function toggleCorpus(id) {
    const next = corpora.includes(id)
      ? corpora.filter((c) => c !== id)
      : [...corpora, id];
    if (next.length > 0) {
      onCorporaChange(next);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey && canSearch && !disabled) {
      e.preventDefault();
      onSearch();
    }
  }

  return (
    <section className="input-section">
      <label htmlFor="term" className="input-label">
        Појам за претрагу (унос на ћирилици)
      </label>
      <input
        id="term"
        type="text"
        className="term-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={modePlaceholder(mode)}
        disabled={disabled}
        autoComplete="off"
      />

      <div className="mode-row">
        <span className="mode-row-label">Начин претраге</span>
        <div className="mode-chips" role="group" aria-label="Начин претраге">
          {SEARCH_MODES.map((m) => {
            const active = mode === m.id;
            return (
              <button
                key={m.id}
                type="button"
                className={`mode-chip${active ? " mode-chip--active" : ""}`}
                disabled={disabled}
                aria-pressed={active}
                onClick={() => onModeChange(m.id)}
              >
                {m.label}
              </button>
            );
          })}
        </div>
        <p className="mode-hint">{activeMode.hint}</p>
      </div>

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
          className="search-btn"
          onClick={onSearch}
          disabled={disabled || !canSearch}
        >
          Search
        </button>
      </div>
    </section>
  );
}
