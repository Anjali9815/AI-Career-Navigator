import Timeline from "./Timeline.jsx";
import SkillTags from "./SkillTags.jsx";
import { initials, titleCase } from "../utils.js";

export default function MatchCard({ match }) {
  return (
    <article className="match">
      <div className="match__head">
        <div className="match__avatar">{initials(match.name)}</div>
        <div>
          <h3 className="match__name">{titleCase(match.name)}</h3>
          {match.current_role && (
            <p className="match__role">{match.current_role}</p>
          )}
        </div>
      </div>

      {match.why_relevant && <p className="match__why">{match.why_relevant}</p>}

      {match.timeline?.length > 0 && (
        <div className="match__block">
          <p className="mini-label">Their path</p>
          <Timeline steps={match.timeline} />
        </div>
      )}

      {match.skills?.length > 0 && (
        <div className="match__block">
          <p className="mini-label">Relevant skills</p>
          <SkillTags skills={match.skills} />
        </div>
      )}
    </article>
  );
}