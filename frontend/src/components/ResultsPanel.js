"use client";

export default function ResultsPanel({ result, error, loading }) {
  if (loading) return <div className="results results--loading">Analyzing…</div>;
  if (error) return <div className="results results--error">Error: {error}</div>;
  if (!result) return <div className="results results--empty">Results will appear here.</div>;

  return (
    <section className="results">
      <p className="results-message">{result.message}</p>
      {result.matches?.length > 0 && (
        <ul>
          {result.matches.map((m, i) => (
            <li key={i}>
              “{m.input_snippet}” → {m.bible_ref.book} {m.bible_ref.chapter}:{m.bible_ref.verse} ({m.confidence_type})
            </li>
          ))}
        </ul>
      )}
      {(result.summary?.old_testament != null || result.summary?.new_testament != null) && (
        <p className="results-summary">
          OT: {result.summary.old_testament} | NT: {result.summary.new_testament}
        </p>
      )}
    </section>
  );
}
