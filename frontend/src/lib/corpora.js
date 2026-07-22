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
