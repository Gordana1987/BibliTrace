/** Active corpora — keep in sync with backend/config.py ACTIVE_CORPORA + CORPUS_LABELS */
export const ACTIVE_CORPORA = ["dk", "spc"];

/** Short chip labels (match mockup). */
export const CORPUS_LABELS = {
  dk: "Караџић (ијекав)",
  spc: "СПЦ (ијекав)",
};

export function corpusLabel(id) {
  return CORPUS_LABELS[id] || id;
}

/** Absolute field ceiling — keep in sync with backend SEARCH_MAX_TERM_CHARS. */
export const TERM_MAX_CHARS = 200;
export const TERM_MAX_CHARS_MESSAGE =
  "Достигнута максимална дужина (200 карактера) — претражују се појмови, не пасуси.";

/** Soft word nudge for lemma / semantic only (does not block typing). */
export const SOFT_WARN_WORD_LIMIT = 5;
export const SOFT_WARN_MESSAGE =
  "Ово личи на одломак, не појам — покушајте краћу фразу за боље резултате.";

const WORD_RE = /[\p{L}\p{N}_]+/gu;

/** Count word-like tokens (aligned with backend \\w+ tokenizer). */
export function countTermWords(text) {
  if (!text || typeof text !== "string") return 0;
  const matches = text.match(WORD_RE);
  return matches ? matches.length : 0;
}

export function shouldShowSoftLengthWarn(mode, text) {
  return (
    (mode === "lemma" || mode === "semantic") &&
    countTermWords(text) > SOFT_WARN_WORD_LIMIT
  );
}

/** Search modes — keep in sync with backend SearchMode */
export const SEARCH_MODES = [
  {
    id: "semantic",
    label: "Семантичко",
    hint: "Тражи и стихове без исте речи, по сродном значењу и мотиву",
    placeholder: "нпр. опроштај, љубав према непријатељу…",
  },
  {
    id: "lemma",
    label: "Лема",
    hint: "Сви облици речи са истом основом (лице, род, број…)",
    placeholder: "нпр. опростити, милост, ријеч…",
  },
  {
    id: "exact",
    label: "Егзактно",
    hint: "Тачан облик; опционо * (*праштај = префикс, опрост* = суфикс)",
    placeholder: "нпр. опроштај, опрост*, *праштај…",
  },
];

export function modeLabel(id) {
  return SEARCH_MODES.find((m) => m.id === id)?.label || id;
}

export function modePlaceholder(id) {
  return (
    SEARCH_MODES.find((m) => m.id === id)?.placeholder ||
    "Унесите појам…"
  );
}
