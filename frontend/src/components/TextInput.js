"use client";

import {
  ACTIVE_CORPORA,
  CORPUS_LABELS,
  SEARCH_MODES,
  SOFT_WARN_MESSAGE,
  TERM_MAX_CHARS,
  TERM_MAX_CHARS_MESSAGE,
  modePlaceholder,
  shouldShowSoftLengthWarn,
} from "../lib/corpora";
import BookFilter from "./BookFilter";

function SoftWarnIcon() {
  return (
    <svg
      className="term-warn-icon"
      viewBox="0 0 16 16"
      width="14"
      height="14"
      aria-hidden="true"
      focusable="false"
    >
      <path
        fill="currentColor"
        d="M7.14 1.5a1 1 0 0 1 1.72 0l6.02 10.9A1 1 0 0 1 14.02 14H1.98a1 1 0 0 1-.86-1.6L7.14 1.5ZM8 5.75a.75.75 0 0 0-.75.75v2.5a.75.75 0 0 0 1.5 0V6.5A.75.75 0 0 0 8 5.75Zm0 6a.9.9 0 1 0 0-1.8.9.9 0 0 0 0 1.8Z"
      />
    </svg>
  );
}

function MaxCharsIcon() {
  return (
    <span className="term-max-icon" aria-hidden="true">
      ×
    </span>
  );
}

export default function TextInput({
  value,
  onChange,
  mode,
  onModeChange,
  onSearch,
  disabled,
  corpora,
  onCorporaChange,
  books,
  onBooksChange,
}) {
  const canSearch = value.trim() && corpora.length > 0 && books.length > 0;
  const activeMode = SEARCH_MODES.find((m) => m.id === mode) || SEARCH_MODES[0];
  const atMaxChars = value.length >= TERM_MAX_CHARS;
  const softWarn = !atMaxChars && shouldShowSoftLengthWarn(mode, value);

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
        className={`term-input${atMaxChars ? " term-input--at-limit" : ""}`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={modePlaceholder(mode)}
        disabled={disabled}
        autoComplete="off"
        maxLength={TERM_MAX_CHARS}
      />
      {atMaxChars ? (
        <div className="term-max-banner" role="status">
          <MaxCharsIcon />
          <p className="term-max-banner-text">{TERM_MAX_CHARS_MESSAGE}</p>
        </div>
      ) : null}
      {softWarn ? (
        <p className="term-soft-warn" role="status">
          <SoftWarnIcon />
          <span>{SOFT_WARN_MESSAGE}</span>
        </p>
      ) : null}

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

      <BookFilter books={books} onChange={onBooksChange} disabled={disabled} />

      <div className="input-controls">
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
