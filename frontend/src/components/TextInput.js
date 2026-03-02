"use client";

export default function TextInput({ value, onChange, onAnalyze, disabled }) {
  return (
    <section className="input-section">
      <label htmlFor="text">Serbian text to analyze</label>
      <textarea
        id="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Nalepite ili unesite tekst..."
        rows={12}
        disabled={disabled}
      />
      <button
        type="button"
        onClick={onAnalyze}
        disabled={disabled || !value.trim()}
      >
        Analyze
      </button>
    </section>
  );
}
