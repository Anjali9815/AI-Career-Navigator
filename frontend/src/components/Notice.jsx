export default function Notice({ variant = "warn", children }) {
  return <div className={`notice notice--${variant}`}>{children}</div>;
}
