"use client";

import { useEffect, useRef, useState } from "react";

const ALL_CORPORA = ["dk", "bakotic", "spc"];

const CORPUS_LABELS = {
  dk: "Даничић–Караџић",
  bakotic: "Бакотић",
  spc: "СПЦ",
};

function selectionSummary(corpora, allThree) {
  if (allThree || corpora.length === 3) {
    return "Сва три";
  }
  if (corpora.length === 1) {
    return CORPUS_LABELS[corpora[0]];
  }
  return corpora.map((id) => CORPUS_LABELS[id]).join(", ");
}

export default function TextInput({
  value,
  onChange,
  onAnalyze,
  disabled,
  corpora,
  onCorporaChange,
  allThree,
  onAllThreeChange,
}) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);
  const canAnalyze = value.trim() && corpora.length > 0;

  useEffect(() => {
    if (!open) return;
    function onDocClick(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  function toggleCorpus(id) {
    if (allThree) return;
    const next = corpora.includes(id)
      ? corpora.filter((c) => c !== id)
      : [...corpora, id];
    if (next.length > 0) {
      onCorporaChange(next);
      onAllThreeChange(false);
    }
  }

  function handleAllThree(checked) {
    onAllThreeChange(checked);
    if (checked) {
      onCorporaChange([...ALL_CORPORA]);
    }
  }

  return (
    <section className="input-section">
      <label htmlFor="text" className="input-label">
        Serbian text to analyze
      </label>
      <textarea
        id="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Nalepite ili unesite tekst..."
        rows={12}
        disabled={disabled}
      />
      <div className="input-controls">
        <div className="corpus-dropdown-wrap" ref={wrapRef}>
          <span className="corpus-dropdown-label">Корпуси</span>
          <button
            type="button"
            className="corpus-dropdown-btn"
            disabled={disabled}
            aria-expanded={open}
            aria-haspopup="listbox"
            onClick={() => setOpen((v) => !v)}
          >
            <span className="corpus-dropdown-btn-text">
              {selectionSummary(corpora, allThree)}
            </span>
            <span className="corpus-dropdown-chevron" aria-hidden>
              ▾
            </span>
          </button>
          {open && !disabled && (
            <div className="corpus-dropdown-menu" role="listbox">
              <label className="corpus-menu-item corpus-menu-item--all">
                <input
                  type="checkbox"
                  className="corpus-menu-checkbox"
                  checked={allThree}
                  onChange={(e) => handleAllThree(e.target.checked)}
                />
                <span className="corpus-menu-label">Сва три</span>
              </label>
              <div className="corpus-menu-divider" />
              {ALL_CORPORA.map((id) => (
                <label key={id} className="corpus-menu-item">
                  <input
                    type="checkbox"
                    className="corpus-menu-checkbox"
                    checked={corpora.includes(id)}
                    disabled={allThree}
                    onChange={() => toggleCorpus(id)}
                  />
                  <span className="corpus-menu-label">{CORPUS_LABELS[id]}</span>
                </label>
              ))}
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={onAnalyze}
          disabled={disabled || !canAnalyze}
        >
          Analyze
        </button>
      </div>
    </section>
  );
}
