// All backend calls live here, so a URL or endpoint change touches one file.

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function findMatches(query) {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    throw new Error(`Server returned ${res.status}`);
  }

  return res.json();
}
