"use client";

import { corpusLabel, modeLabel } from "../lib/corpora";

function HitCard({ hit, ranking }) {
  const showScore = ranking === "score" && typeof hit.score === "number";
  const showMatchMark = ranking === "biblical_order";

  return (
    <article className="result-card">
      <div className="result-card-top">
        <strong className="result-card-ref">
          {hit.book} {hit.chapter}:{hit.verse}
        </strong>
        {showScore && (
          <span className="result-card-score">{hit.score.toFixed(2)}</span>
        )}
        {showMatchMark && (
          <span className="result-card-match" title="Погодак" aria-label="Погодак">
            ✓
          </span>
        )}
      </div>
      <p className="result-card-text">{hit.text || ""}</p>
    </article>
  );
}

function CorpusPanel({
  corpusId,
  corpusResult,
  mode,
  loadingMore,
  onLoadMore,
  onTrySemantic,
}) {
  const hits = corpusResult?.hits || [];
  const total = corpusResult?.total ?? 0;
  const ranking = corpusResult?.ranking || "biblical_order";
  const hasMore = hits.length < total;
  const empty = total === 0;
  const offerSemantic = empty && mode !== "semantic";

  return (
    <section className="results-panel-column">
      <header className="results-panel-header">
        <h3 className="results-panel-title">{corpusLabel(corpusId)}</h3>
        <span className="model-chip">{modeLabel(mode)}</span>
      </header>

      {total > 0 && (
        <p className="results-count">
          {total} резултата — приказано {hits.length}
          {ranking === "biblical_order" && (
            <span className="results-count-note"> · редослед по Библији</span>
          )}
          {ranking === "score" && (
            <span className="results-count-note"> · по сродности</span>
          )}
        </p>
      )}

      <div className="results-panel-list">
        {hits.length > 0 ? (
          hits.map((hit, index) => (
            <HitCard
              key={`${corpusId}-${hit.book}-${hit.chapter}-${hit.verse}-${index}`}
              hit={hit}
              ranking={ranking}
            />
          ))
        ) : (
          <div className="results-corpus-empty-block">
            <p className="results-corpus-empty">Нема погодака.</p>
            {offerSemantic && (
              <button
                type="button"
                className="try-semantic-btn"
                onClick={onTrySemantic}
              >
                Пробај Семантичко
              </button>
            )}
          </div>
        )}
      </div>

      {hasMore && (
        <div className="results-panel-actions">
          <button
            type="button"
            className="load-more-btn"
            onClick={() => onLoadMore(corpusId)}
            disabled={loadingMore}
          >
            {loadingMore ? "Учитавам…" : "Учитај +20"}
          </button>
        </div>
      )}
    </section>
  );
}

export default function ResultsPanel({
  result,
  error,
  loading,
  loadingMoreByCorpus,
  selectedCorpora,
  onLoadMore,
  onTrySemantic,
}) {
  if (loading) return <div className="results results--loading">Претражујем…</div>;
  if (error) return <div className="results results--error">Грешка: {error}</div>;
  if (!result) {
    return (
      <div className="results results--empty">
        Резултати ће се појавити овде — сваки корпус у свом панелу.
      </div>
    );
  }

  const byCorpus = result.results_by_corpus || {};
  const corpusIds = selectedCorpora?.length
    ? selectedCorpora
    : Object.keys(byCorpus);

  return (
    <section className="results">
      {result.message && <p className="results-message">{result.message}</p>}
      <div className="results-board-wrap">
        <div className="results-board">
          {corpusIds.map((corpusId) => (
            <CorpusPanel
              key={corpusId}
              corpusId={corpusId}
              corpusResult={byCorpus[corpusId]}
              mode={result.mode}
              loadingMore={Boolean(loadingMoreByCorpus?.[corpusId])}
              onLoadMore={onLoadMore}
              onTrySemantic={onTrySemantic}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
