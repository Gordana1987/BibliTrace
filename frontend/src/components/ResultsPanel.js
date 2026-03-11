"use client";

export default function ResultsPanel({
  result,
  error,
  loading,
  loadingLabse,
  onCompareWithLabse,
}) {
  if (loading) return <div className="results results--loading">Analyzing…</div>;
  if (error) return <div className="results results--error">Error: {error}</div>;
  if (!result) return <div className="results results--empty">Results will appear here.</div>;

  const hasLabse = result.labse_matches != null;
  const canCompare = result.matches?.length > 0 && !hasLabse;

  return (
    <section className="results">
      <p className="results-message">{result.message}</p>
      {result.matches?.length > 0 && (
        <div className="results-box">
          <h3 className="results-box-title">Qwen3</h3>
          <ul>
            {result.matches.map((m, i) => {
              const verseText = m.bible_ref?.text || "";
              return (
                <li key={i}>
                  &quot;{verseText}&quot; — {m.bible_ref.book} {m.bible_ref.chapter}:{m.bible_ref.verse}{" "}
                  ({m.confidence_type})
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {canCompare && (
        <button
          type="button"
          className="compare-btn"
          onClick={onCompareWithLabse}
          disabled={loadingLabse}
        >
          {loadingLabse ? "Loading LaBSE…" : "Compare with LaBSE"}
        </button>
      )}
      {hasLabse && (
        <div className="results-box results-box--labse">
          <h3 className="results-box-title">LaBSE</h3>
          <ul>
            {result.labse_matches.map((m, i) => {
              const verseText = m.bible_ref?.text || "";
              return (
                <li key={i}>
                  &quot;{verseText}&quot; — {m.bible_ref.book} {m.bible_ref.chapter}:{m.bible_ref.verse}{" "}
                  ({m.confidence_type})
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {(result.summary?.old_testament != null || result.summary?.new_testament != null) && (
        <p className="results-summary">
          OT: {result.summary.old_testament} | NT: {result.summary.new_testament}
        </p>
      )}
    </section>
  );
}
