import Notice from "./Notice.jsx";
import DirectionPanel from "./DirectionPanel.jsx";
import MatchList from "./MatchList.jsx";
import NextSteps from "./NextSteps.jsx";

export default function ResultView({ result }) {
  if (result.no_match) {
    return (
      <div className="panel">
        <p className="muted-text">
          {result.message ||
            "No matching career profiles were found in our database to illustrate this path."}
        </p>
      </div>
    );
  }

  return (
    <div>
      {result.low_confidence && (
        <Notice variant="warn">
          This is the closest profile we have, not a strong match. Try describing
          your background in more detail.
        </Notice>
      )}

      <DirectionPanel direction={result.direction} />
      <MatchList matches={result.matches} />
      <NextSteps steps={result.next_steps} />
    </div>
  );
}
