/** Active corpora — keep in sync with backend/config.py ACTIVE_CORPORA + CORPUS_LABELS */
export const ACTIVE_CORPORA = ["dk", "spc"];

/** Short chip labels (match mockup). */
export const CORPUS_LABELS = {
  dk: "Даничић (ДК)",
  spc: "СПЦ (НЗ)",
};

export function corpusLabel(id) {
  return CORPUS_LABELS[id] || id;
}
