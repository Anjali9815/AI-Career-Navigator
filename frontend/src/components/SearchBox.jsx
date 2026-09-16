export default function SearchBox({ query, onChange, onSubmit, loading }) {
  const disabled = loading || !query.trim();

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  }

  return (
    <div className="search">
      <textarea
        className="search__input"
        value={query}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="For example: I am finishing a CS degree and want to work in machine learning"
        rows={4}
      />
      <div className="search__actions">
        <span className="search__hint">
          Enter to search, Shift and Enter for a new line
        </span>
        <button className="search__button" onClick={onSubmit} disabled={disabled}>
          {loading ? "Searching..." : "Find my match"}
        </button>
      </div>
    </div>
  );
}
