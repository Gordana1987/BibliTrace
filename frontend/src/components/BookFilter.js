"use client";

import { useEffect, useId, useRef, useState } from "react";
import {
  NT_ALL_BOOKS,
  NT_BOOK_TREE,
  collectNodeBooks,
  nodeBookCount,
} from "../lib/ntBooks";

function selectionState(selectedSet, books) {
  const n = books.filter((b) => selectedSet.has(b)).length;
  if (n === 0) return "none";
  if (n === books.length) return "all";
  return "partial";
}

function CascadeCheckbox({ id, label, count, state, disabled, onToggle }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.indeterminate = state === "partial";
    }
  }, [state]);

  return (
    <label className="book-check" htmlFor={id}>
      <input
        ref={ref}
        id={id}
        type="checkbox"
        className="book-check-input"
        checked={state === "all"}
        disabled={disabled}
        onChange={onToggle}
      />
      <span className="book-check-label">
        {label}
        {count != null ? (
          <span className="book-check-count"> ({count})</span>
        ) : null}
      </span>
    </label>
  );
}

function BookNode({ node, selectedSet, disabled, onToggleBooks, depth = 0 }) {
  const uid = useId();
  const books = collectNodeBooks(node);
  const state = selectionState(selectedSet, books);
  const count = nodeBookCount(node);
  const isLeafGroup =
    Array.isArray(node.books) &&
    node.books.length === 1 &&
    node.books[0] === node.label;

  function toggleGroup() {
    onToggleBooks(books, state !== "all");
  }

  return (
    <div
      className={`book-node book-node--depth-${depth}${
        isLeafGroup ? " book-node--leaf-group" : ""
      }`}
    >
      <CascadeCheckbox
        id={`${uid}-${node.id}`}
        label={node.label}
        count={count}
        state={state}
        disabled={disabled}
        onToggle={toggleGroup}
      />

      {!isLeafGroup && node.books ? (
        <div
          className={`book-leaf-grid${
            node.books.length <= 4 ? " book-leaf-grid--row" : ""
          }`}
        >
          {node.books.map((book) => {
            const checked = selectedSet.has(book);
            return (
              <label key={book} className="book-check book-check--leaf">
                <input
                  type="checkbox"
                  className="book-check-input"
                  checked={checked}
                  disabled={disabled}
                  onChange={() => onToggleBooks([book], !checked)}
                />
                <span className="book-check-label">{book}</span>
              </label>
            );
          })}
        </div>
      ) : null}

      {node.children
        ? node.children.map((child) => (
            <BookNode
              key={child.id}
              node={child}
              selectedSet={selectedSet}
              disabled={disabled}
              onToggleBooks={onToggleBooks}
              depth={depth + 1}
            />
          ))
        : null}
    </div>
  );
}

export default function BookFilter({ books, onChange, disabled }) {
  const [open, setOpen] = useState(false);
  const selectedSet = new Set(books);
  const selectedCount = books.length;
  const total = NT_ALL_BOOKS.length;
  const summary =
    selectedCount === total
      ? `Све (${total}/${total})`
      : `${selectedCount}/${total}`;

  function toggleBooks(names, select) {
    const next = new Set(books);
    for (const name of names) {
      if (select) next.add(name);
      else next.delete(name);
    }
    onChange([...NT_ALL_BOOKS].filter((b) => next.has(b)));
  }

  function selectAll() {
    onChange([...NT_ALL_BOOKS]);
  }

  function clearAll() {
    onChange([]);
  }

  return (
    <div className="book-filter">
      <div className="book-filter-header">
        <span className="book-filter-label">Књиге</span>
        <button
          type="button"
          className={`book-filter-trigger${open ? " book-filter-trigger--open" : ""}`}
          disabled={disabled}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {summary}
          <span className="book-filter-chevron" aria-hidden>
            ▾
          </span>
        </button>
      </div>

      {open ? (
        <div className="book-filter-panel">
          {NT_BOOK_TREE.map((node) => (
            <BookNode
              key={node.id}
              node={node}
              selectedSet={selectedSet}
              disabled={disabled}
              onToggleBooks={toggleBooks}
            />
          ))}
          <div className="book-filter-actions">
            <button
              type="button"
              className="book-filter-action"
              disabled={disabled || selectedCount === total}
              onClick={selectAll}
            >
              Изабери све
            </button>
            <button
              type="button"
              className="book-filter-action"
              disabled={disabled || selectedCount === 0}
              onClick={clearAll}
            >
              Поништи све
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
