import { useState, useEffect, useRef } from "react";

const API = "http://localhost:8000";

const MODE_OPTIONS = [
  { value: "resume", label: "Profiles" },
  { value: "job",    label: "Jobs" },
  { value: "match",  label: "Match" },
  { value: "all",    label: "All" },
];

function ScoreBadge({ score }) {
  const pct = Math.round((1 - score) * 100);
  const color = pct >= 70 ? "#4ade80" : pct >= 50 ? "#facc15" : "#f87171";
  return (
    <span style={{
      fontSize: 11, fontWeight: 600, letterSpacing: 1,
      color, border: `1px solid ${color}33`,
      borderRadius: 4, padding: "2px 7px", background: `${color}11`
    }}>{pct}%</span>
  );
}

function SourceCard({ source, rank, score, text, page }) {
  const [open, setOpen] = useState(false);
  return (
    <div onClick={() => setOpen(o => !o)} style={{
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.09)",
      borderRadius: 10, padding: "10px 14px", cursor: "pointer",
      transition: "background 0.2s",
      marginBottom: 8,
    }}
    onMouseEnter={e => e.currentTarget.style.background = "rgba(255,255,255,0.08)"}
    onMouseLeave={e => e.currentTarget.style.background = "rgba(255,255,255,0.04)"}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{
            width: 20, height: 20, borderRadius: "50%",
            background: "rgba(139,92,246,0.3)", border: "1px solid rgba(139,92,246,0.5)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 10, color: "#a78bfa", fontWeight: 700, flexShrink: 0
          }}>{rank}</span>
          <span style={{ fontSize: 12, color: "#e2e8f0", fontWeight: 500 }}>{source}</span>
          <span style={{ fontSize: 10, color: "#64748b" }}>p{page}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <ScoreBadge score={score} />
          <span style={{ fontSize: 10, color: "#475569" }}>{open ? "▲" : "▼"}</span>
        </div>
      </div>
      {open && (
        <div style={{
          marginTop: 10, fontSize: 11, color: "#94a3b8",
          lineHeight: 1.6, borderTop: "1px solid rgba(255,255,255,0.06)",
          paddingTop: 10, fontFamily: "monospace"
        }}>{text}</div>
      )}
    </div>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div style={{
      display: "flex", justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: 16, animation: "fadeUp 0.3s ease"
    }}>
      {!isUser && (
        <div style={{
          width: 28, height: 28, borderRadius: "50%", flexShrink: 0,
          background: "linear-gradient(135deg, #7c3aed, #4f46e5)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 12, marginRight: 10, marginTop: 2
        }}>✦</div>
      )}
      <div style={{ maxWidth: "75%" }}>
        <div style={{
          background: isUser
            ? "linear-gradient(135deg, rgba(124,58,237,0.4), rgba(79,70,229,0.4))"
            : "rgba(255,255,255,0.06)",
          border: `1px solid ${isUser ? "rgba(139,92,246,0.3)" : "rgba(255,255,255,0.08)"}`,
          backdropFilter: "blur(12px)",
          borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
          padding: "12px 16px",
          fontSize: 14, color: "#e2e8f0", lineHeight: 1.7,
          whiteSpace: "pre-wrap"
        }}>
          {msg.content}
        </div>
        {msg.sources && msg.sources.length > 0 && (
          <div style={{ marginTop: 6, fontSize: 11, color: "#64748b" }}>
            Sources: {msg.sources.join(", ")}
          </div>
        )}
        {msg.chunks && msg.chunks.length > 0 && (
          <div style={{ marginTop: 10 }}>
            {msg.chunks.map((c, i) => (
              <SourceCard key={i} rank={c.rank} source={c.source}
                score={c.score} text={c.text} page={c.page} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ProfileCard({ doc, index }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 12, padding: "14px 16px",
      backdropFilter: "blur(10px)",
      animation: `fadeUp 0.4s ease ${index * 0.05}s both`,
      transition: "transform 0.2s, background 0.2s",
      cursor: "default"
    }}
    onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.background = "rgba(255,255,255,0.08)"; }}
    onMouseLeave={e => { e.currentTarget.style.transform = "translateY(0)"; e.currentTarget.style.background = "rgba(255,255,255,0.04)"; }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{
          width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
          background: `hsl(${(index * 47) % 360}, 60%, 45%)`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 13, fontWeight: 700, color: "#fff"
        }}>{doc.filename.slice(0, 1).toUpperCase()}</div>
        <div>
          <div style={{ fontSize: 12, color: "#e2e8f0", fontWeight: 500 }}>
            {doc.filename.replace(".pdf", "")}
          </div>
          <div style={{ fontSize: 10, color: "#64748b", marginTop: 1 }}>
            {doc.doc_type}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages]   = useState([]);
  const [input, setInput]         = useState("");
  const [mode, setMode]           = useState("resume");
  const [loading, setLoading]     = useState(false);
  const [sources, setSources]     = useState([]);
  const [stats, setStats]         = useState(null);
  const [sidebarTab, setSidebarTab] = useState("profiles");
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  useEffect(() => {
    fetch(`${API}/sources`)
      .then(r => r.json())
      .then(data => {
        setSources(data.documents || []);
        setStats({ total: data.total_documents, chunks: data.total_chunks });
      })
      .catch(() => {});

    setMessages([{
      role: "assistant",
      content: "Hi! I'm your AI Career Navigator. I've loaded all the LinkedIn profiles into my knowledge base.\n\nTry asking:\n• \"Who has the strongest Python and ML experience?\"\n• \"Which candidates have AWS or cloud experience?\"\n• \"Find someone with NLP and production experience\"",
    }]);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const query = input.trim();
    setInput("");
    setMessages(m => [...m, { role: "user", content: query }]);
    setLoading(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, mode })
      });
      const data = await res.json();

      if (!res.ok) {
        setMessages(m => [...m, { role: "assistant", content: `Error: ${data.detail}` }]);
      } else {
        const searchRes = await fetch(`${API}/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, doc_type: mode === "all" ? null : mode === "resume" ? "resume" : "job_description", k: 4 })
        });
        const searchData = await searchRes.json();

        setMessages(m => [...m, {
          role: "assistant",
          content: data.answer,
          sources: data.sources,
          chunks: searchData.chunks || []
        }]);
      }
    } catch {
      setMessages(m => [...m, { role: "assistant", content: "Connection error — is the FastAPI server running?" }]);
    }
    setLoading(false);
  };

  const handleKey = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } };

  return (
    <div style={{
      minHeight: "100vh", background: "#0a0a0f",
      backgroundImage: "radial-gradient(ellipse at 20% 20%, rgba(124,58,237,0.15) 0%, transparent 50%), radial-gradient(ellipse at 80% 80%, rgba(79,70,229,0.1) 0%, transparent 50%)",
      display: "flex", flexDirection: "column", fontFamily: "'DM Sans', system-ui, sans-serif",
      color: "#e2e8f0"
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.3); border-radius: 2px; }
        @keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        textarea:focus { outline: none; }
        textarea { resize: none; }
      `}</style>

      {/* ── HEADER ── */}
      <div style={{
        padding: "16px 24px", display: "flex", alignItems: "center",
        justifyContent: "space-between",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        background: "rgba(10,10,15,0.8)", backdropFilter: "blur(20px)",
        position: "sticky", top: 0, zIndex: 10
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "linear-gradient(135deg, #7c3aed, #4f46e5)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, boxShadow: "0 0 20px rgba(124,58,237,0.4)"
          }}>✦</div>
          <div>
            <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 16, fontWeight: 800, letterSpacing: -0.5 }}>
              Career Navigator
            </div>
            <div style={{ fontSize: 11, color: "#64748b" }}>RAG · Gemini · ChromaDB</div>
          </div>
        </div>

        {/* Mode selector */}
        <div style={{ display: "flex", gap: 4, background: "rgba(255,255,255,0.04)", borderRadius: 10, padding: 4 }}>
          {MODE_OPTIONS.map(m => (
            <button key={m.value} onClick={() => setMode(m.value)} style={{
              padding: "6px 14px", borderRadius: 7, border: "none", cursor: "pointer",
              fontSize: 12, fontWeight: 500, transition: "all 0.2s",
              background: mode === m.value ? "linear-gradient(135deg, #7c3aed, #4f46e5)" : "transparent",
              color: mode === m.value ? "#fff" : "#64748b",
              boxShadow: mode === m.value ? "0 2px 12px rgba(124,58,237,0.4)" : "none"
            }}>{m.label}</button>
          ))}
        </div>

        {stats && (
          <div style={{ display: "flex", gap: 16 }}>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#a78bfa", fontFamily: "'Syne', sans-serif" }}>{stats.total}</div>
              <div style={{ fontSize: 10, color: "#475569" }}>profiles</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#818cf8", fontFamily: "'Syne', sans-serif" }}>{stats.chunks}</div>
              <div style={{ fontSize: 10, color: "#475569" }}>chunks</div>
            </div>
          </div>
        )}
      </div>

      {/* ── MAIN LAYOUT ── */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden", height: "calc(100vh - 69px)" }}>

        {/* ── SIDEBAR ── */}
        <div style={{
          width: 280, flexShrink: 0, borderRight: "1px solid rgba(255,255,255,0.06)",
          display: "flex", flexDirection: "column",
          background: "rgba(255,255,255,0.01)"
        }}>
          {/* Sidebar tabs */}
          <div style={{ display: "flex", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
            {["profiles", "tips"].map(tab => (
              <button key={tab} onClick={() => setSidebarTab(tab)} style={{
                flex: 1, padding: "12px 0", border: "none", cursor: "pointer",
                background: "transparent", fontSize: 11, fontWeight: 600,
                letterSpacing: 0.5, textTransform: "uppercase",
                color: sidebarTab === tab ? "#a78bfa" : "#475569",
                borderBottom: sidebarTab === tab ? "2px solid #7c3aed" : "2px solid transparent",
                transition: "all 0.2s"
              }}>{tab}</button>
            ))}
          </div>

          {/* Sidebar content */}
          <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
            {sidebarTab === "profiles" ? (
              <>
                <div style={{ fontSize: 11, color: "#475569", marginBottom: 12, letterSpacing: 0.5, textTransform: "uppercase" }}>
                  Loaded profiles
                </div>
                {sources.length === 0 ? (
                  <div style={{ fontSize: 12, color: "#334155", textAlign: "center", marginTop: 40 }}>
                    No profiles loaded
                  </div>
                ) : (
                  sources.map((doc, i) => <ProfileCard key={i} doc={doc} index={i} />)
                )}
              </>
            ) : (
              <>
                <div style={{ fontSize: 11, color: "#475569", marginBottom: 12, letterSpacing: 0.5, textTransform: "uppercase" }}>
                  Sample queries
                </div>
                {[
                  "Who has the strongest Python and ML experience?",
                  "Which candidates have AWS or cloud platform experience?",
                  "Find someone with NLP or LLM production experience",
                  "Who has the most years of software engineering experience?",
                  "Which profiles mention FastAPI or backend development?",
                  "Who has experience with data pipelines or ETL?",
                ].map((tip, i) => (
                  <div key={i} onClick={() => { setInput(tip); inputRef.current?.focus(); }} style={{
                    fontSize: 12, color: "#94a3b8", padding: "10px 12px",
                    borderRadius: 8, cursor: "pointer", marginBottom: 6,
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.06)",
                    lineHeight: 1.5, transition: "all 0.2s"
                  }}
                  onMouseEnter={e => { e.currentTarget.style.background = "rgba(124,58,237,0.1)"; e.currentTarget.style.borderColor = "rgba(124,58,237,0.3)"; }}
                  onMouseLeave={e => { e.currentTarget.style.background = "rgba(255,255,255,0.03)"; e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)"; }}
                  >{tip}</div>
                ))}
              </>
            )}
          </div>
        </div>

        {/* ── CHAT AREA ── */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

          {/* Messages */}
          <div style={{ flex: 1, overflowY: "auto", padding: "24px 32px" }}>
            {messages.map((msg, i) => <Message key={i} msg={msg} />)}
            {loading && (
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                <div style={{
                  width: 28, height: 28, borderRadius: "50%",
                  background: "linear-gradient(135deg, #7c3aed, #4f46e5)",
                  display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12
                }}>✦</div>
                <div style={{ display: "flex", gap: 4 }}>
                  {[0,1,2].map(i => (
                    <div key={i} style={{
                      width: 6, height: 6, borderRadius: "50%",
                      background: "#7c3aed",
                      animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite`
                    }}/>
                  ))}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input area */}
          <div style={{
            padding: "16px 32px 24px",
            borderTop: "1px solid rgba(255,255,255,0.06)",
            background: "rgba(10,10,15,0.6)", backdropFilter: "blur(20px)"
          }}>
            <div style={{
              display: "flex", gap: 12, alignItems: "flex-end",
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 14, padding: "12px 16px",
              backdropFilter: "blur(10px)",
              transition: "border-color 0.2s",
            }}
            onFocus={() => {}}
            >
              <textarea
                ref={inputRef}
                rows={1}
                value={input}
                onChange={e => { setInput(e.target.value); e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px"; }}
                onKeyDown={handleKey}
                placeholder={`Ask about ${mode === "resume" ? "candidate profiles" : mode === "job" ? "job descriptions" : "all documents"}...`}
                style={{
                  flex: 1, background: "transparent", border: "none",
                  color: "#e2e8f0", fontSize: 14, lineHeight: 1.5,
                  fontFamily: "'DM Sans', sans-serif",
                  maxHeight: 120, overflowY: "auto"
                }}
              />
              <button onClick={sendMessage} disabled={loading || !input.trim()} style={{
                width: 36, height: 36, borderRadius: 9, border: "none",
                cursor: loading || !input.trim() ? "not-allowed" : "pointer",
                background: loading || !input.trim()
                  ? "rgba(255,255,255,0.06)"
                  : "linear-gradient(135deg, #7c3aed, #4f46e5)",
                color: "#fff", fontSize: 16, flexShrink: 0,
                transition: "all 0.2s",
                boxShadow: loading || !input.trim() ? "none" : "0 2px 12px rgba(124,58,237,0.4)"
              }}>↑</button>
            </div>
            <div style={{ fontSize: 10, color: "#334155", marginTop: 8, textAlign: "center" }}>
              Enter to send · Shift+Enter for new line · Mode: {mode}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
