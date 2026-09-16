export default function Timeline({ steps }) {
  if (!steps?.length) return null;

  return (
    <ol className="timeline">
      {steps.map((step, i) => (
        <li
          key={i}
          className={`timeline__item timeline__item--${step.kind || "experience"}`}
        >
          <span className="timeline__label">{step.label}</span>
          {step.dates && <span className="timeline__dates">{step.dates}</span>}
        </li>
      ))}
    </ol>
  );
}