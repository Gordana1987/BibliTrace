"use client";

export default function TextInput({ value, onChange, onAnalyze, disabled, version, onVersionChange }) {
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
      <div className="input-controls">
        <select
          className="version-select"
          value={version}
          onChange={(e) => onVersionChange(e.target.value)}
          disabled={disabled}
        >
          <option value="dk">Daničić–Karadžić</option>
          <option value="bakotic">Bakotić</option>
          <option value="both">Both</option>
        </select>
        <button
          type="button"
          onClick={onAnalyze}
          disabled={disabled || !value.trim()}
        >
          Analyze
        </button>
      </div>
    </section>
  );
}
