"use client";

import { corpusLabel } from "../lib/corpora";

function MatchCard({ match }) {
  const verseText = match.bible_ref?.text || "";
  const score =
    typeof match.score === "number" ? match.score.toFixed(2) : match.score ?? "";

  return (
    <article className="result-card">
      <div className="result-card-top">
        <strong className="result-card-ref">
          {match.bible_ref.book} {match.bible_ref.chapter}:{match.bible_ref.verse}
        </strong>
        <span className="result-card-score">{score}</span>
      </div>
      <p className="result-card-text">{verseText}</p>
    </article>
  );
}

function CorpusPanel({
  corpusId,
  qwenMatches,
  labseMatches,
  loadingLabse,
  onLoadLabse,
}) {
  const hasQwen = qwenMatches?.length > 0;
  const hasLabse = labseMatches?.length > 0;

  return (
    <section className="results-panel-column">
      <header className="results-panel-header">
        <h3 className="results-panel-title">{corpusLabel(corpusId)}</h3>
        <span className="model-chip">Qwen3</span>
      </header>

      <div className="results-panel-list">
        {hasQwen ? (
          qwenMatches.map((match, index) => (
            <MatchCard key={`${corpusId}-qwen-${index}`} match={match} />
          ))
        ) : (
          <p className="results-corpus-empty">Нема погодака.</p>
        )}
      </div>

      <div className="results-panel-actions">
        <button
          type="button"
          className="labse-load-btn"
          onClick={() => onLoadLabse(corpusId)}
          disabled={loadingLabse || hasLabse}
        >
          {hasLabse
            ? "LaBSE учитан"
            : loadingLabse
              ? "Учитавам LaBSE…"
              : "Учитај +20 (LaBSE)"}
        </button>
      </div>

      {hasLabse && (
        <div className="results-panel-labse">
          <div className="results-panel-subhead">LaBSE резултати</div>
          <div className="results-panel-list">
            {labseMatches.map((match, index) => (
              <MatchCard key={`${corpusId}-labse-${index}`} match={match} />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export default function ResultsPanel({
  result,
  error,
  loading,
  loadingLabseByCorpus,
  selectedCorpora,
  onLoadLabse,
}) {
  if (loading) return <div className="results results--loading">Анализирам…</div>;
  if (error) return <div className="results results--error">Грешка: {error}</div>;
  if (!result) {
    return (
      <div className="results results--empty">
        Резултати ће се појавити овде — сваки корпус у свом панелу.
      </div>
    );
  }

  const byCorpus =
    result.matches_by_corpus && Object.keys(result.matches_by_corpus).length > 0
      ? result.matches_by_corpus
      : result.matches?.length
        ? { dk: result.matches }
        : {};

  const labseByCorpus =
    result.labse_matches_by_corpus && Object.keys(result.labse_matches_by_corpus).length > 0
      ? result.labse_matches_by_corpus
      : result.labse_matches?.length
        ? { dk: result.labse_matches }
        : null;

  const hasAny = Object.values(byCorpus).some((matches) => matches?.length > 0);
  const corpusIds = selectedCorpora?.length ? selectedCorpora : Object.keys(byCorpus);

  return (
    <section className="results">
      {result.message && <p className="results-message">{result.message}</p>}
      {hasAny && (
        <div className="results-board-wrap">
          <div className="results-board">
            {corpusIds.map((corpusId) => (
              <CorpusPanel
                key={corpusId}
                corpusId={corpusId}
                qwenMatches={byCorpus[corpusId] || []}
                labseMatches={labseByCorpus?.[corpusId] || []}
                loadingLabse={Boolean(loadingLabseByCorpus?.[corpusId])}
                onLoadLabse={onLoadLabse}
              />
            ))}
          </div>
        </div>
      )}
      {!hasAny && (
        <p className="results-corpus-empty">Нема лексичких кандидата. Унесите ћирилични текст.</p>
      )}
    </section>
  );
}
