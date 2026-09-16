export default function SkillTags({ skills }) {
  if (!skills?.length) return null;

  return (
    <div className="tags">
      {skills.map((s, i) => (
        <span key={i} className="tag">
          {s}
        </span>
      ))}
    </div>
  );
}
