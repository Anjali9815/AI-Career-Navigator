import { useState } from "react";

import Header from "./components/Header.jsx";
import SearchBox from "./components/SearchBox.jsx";
import Examples from "./components/Examples.jsx";
import Notice from "./components/Notice.jsx";
import ResultView from "./components/ResultView.jsx";

import { findMatches } from "./api.js";

export default function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit() {
    const trimmed = query.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await findMatches(trimmed);
      setResult(data);
    } catch (err) {
      setError("Could not reach the server. Is the API running on port 8000?");
    } finally {
      setLoading(false);
    }
  }

  const idle = !result && !loading && !error;

  return (
    <div className="page">
      <div className="container">
        <Header />

        <SearchBox
          query={query}
          onChange={setQuery}
          onSubmit={handleSubmit}
          loading={loading}
        />

        {idle && <Examples onPick={setQuery} />}

        {loading && (
          <div className="panel">
            <p className="muted-text">
              Searching profiles and building your match...
            </p>
          </div>
        )}

        {error && <Notice variant="error">{error}</Notice>}

        {result && <ResultView result={result} />}

        <footer className="footer">
          Matched against real alumni profiles. Not AI generated suggestions.
        </footer>
      </div>
    </div>
  );
}