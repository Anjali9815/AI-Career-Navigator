import { EXAMPLE_QUERIES } from "../constants.js";

export default function Examples({ onPick }) {
  return (
    <div className="examples">
      <p className="section-label">Try one of these</p>
      {EXAMPLE_QUERIES.map((ex) => (
        <button key={ex} className="examples__chip" onClick={() => onPick(ex)}>
          {ex}
        </button>
      ))}
    </div>
  );
}
