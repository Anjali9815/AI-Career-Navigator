export default function NextSteps({ steps }) {
  if (!steps?.length) return null;

  return (
    <section className="panel">
      <p className="section-label">Your next steps</p>
      <ol className="steps">
        {steps.map((s, i) => (
          <li key={i} className="steps__item">
            <span className="steps__number">{i + 1}</span>
            <span className="steps__text">{s}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
