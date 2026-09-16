import MatchCard from "./MatchCard.jsx";

export default function MatchList({ matches }) {
  if (!matches?.length) return null;

  const label =
    matches.length === 1 ? "1 matching path" : `${matches.length} matching paths`;

  return (
    <section className="matches">
      <p className="section-label">{label}</p>
      {matches.map((m, i) => (
        <MatchCard key={`${m.name}-${i}`} match={m} />
      ))}
    </section>
  );
}
