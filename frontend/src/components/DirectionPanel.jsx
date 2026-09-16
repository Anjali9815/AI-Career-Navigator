export default function DirectionPanel({ direction }) {
  if (!direction) return null;

  return (
    <section className="panel">
      <p className="section-label">Where this points</p>
      <p className="direction__text">{direction}</p>
    </section>
  );
}
