const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

/**
 * Concept search: POST /api/search
 * @param {{ term: string, mode: 'exact'|'lemma'|'semantic', corpora: string[], offset?: number, limit?: number }} params
 */
export async function searchConcept({
  term,
  mode,
  corpora = ["dk"],
  offset = 0,
  limit = 20,
}) {
  const res = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ term, mode, corpora, offset, limit }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Backend not healthy");
  return res.json();
}
