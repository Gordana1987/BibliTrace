/** NZ book hierarchy — keep Pouke names in sync with backend/config.py DK_NT_BOOK_ORDER */

export const NT_GOSPELS = ["Матеј", "Марко", "Лука", "Јован"];

export const NT_PAULINE = [
  "Римљанима",
  "1. Коринћанима",
  "2. Коринћанима",
  "Галатима",
  "Ефешанима",
  "Филипљанима",
  "Колошанима",
  "1. Солуњанима",
  "2. Солуњанима",
  "1. Тимотеју",
  "2. Тимотеју",
  "Титу",
  "Филимону",
  "Јеврејима",
];

export const NT_GENERAL = [
  "Јаковљева",
  "1. Петрова",
  "2. Петрова",
  "1. Јованова",
  "2. Јованова",
  "3. Јованова",
  "Јудина",
];

/** Flat list of all 27 NT books in canonical order. */
export const NT_ALL_BOOKS = [
  ...NT_GOSPELS,
  "Дела апостолска",
  ...NT_PAULINE,
  ...NT_GENERAL,
  "Откривење",
];

/**
 * Tree for cascade UI.
 * - `books`: leaf names (Pouke)
 * - `children`: nested groups (only Посланице)
 * - Single-book groups (Дела, Откривење) omit a nested grid when label === books[0]
 */
export const NT_BOOK_TREE = [
  {
    id: "gospels",
    label: "Јеванђеља",
    books: NT_GOSPELS,
  },
  {
    id: "acts",
    label: "Дела апостолска",
    books: ["Дела апостолска"],
  },
  {
    id: "epistles",
    label: "Посланице",
    children: [
      {
        id: "pauline",
        label: "Павлове посланице",
        books: NT_PAULINE,
      },
      {
        id: "general",
        label: "Саборне посланице",
        books: NT_GENERAL,
      },
    ],
  },
  {
    id: "revelation",
    label: "Откривење",
    books: ["Откривење"],
  },
];

export function collectNodeBooks(node) {
  if (node.books) return [...node.books];
  if (node.children) return node.children.flatMap(collectNodeBooks);
  return [];
}

export function nodeBookCount(node) {
  return collectNodeBooks(node).length;
}
