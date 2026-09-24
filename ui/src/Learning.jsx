import { useCallback, useEffect, useState } from "react";
import "./Learning.css";

function badgeClass(severity) {
  if (severity === "critical" || severity === "high") return `sev ${severity}`;
  if (severity === "low") return "sev low";
  return "sev medium";
}

function LessonCard({ lesson, onApprove, onReject }) {
  return (
    <article className="lesson-card">
      <div className="lesson-meta">
        <span className={badgeClass(lesson.severity)}>{lesson.severity}</span>
        {(lesson.tags || []).map((tag) => (
          <span key={tag} className="tag">
            {tag}
          </span>
        ))}
        <span className="tag">{lesson.agent_id}</span>
        <span className="confidence">{Number(lesson.confidence || 0).toFixed(2)}</span>
      </div>
      <p className="lesson-text">{lesson.lesson}</p>
      {(lesson.files || []).length > 0 && (
        <p className="lesson-files">{lesson.files.join(" · ")}</p>
      )}
      {onApprove && (
        <div className="lesson-actions">
          <button type="button" className="approve" onClick={() => onApprove(lesson.lesson_id)}>
            Approve
          </button>
          <button type="button" className="reject" onClick={() => onReject(lesson.lesson_id)}>
            Reject
          </button>
        </div>
      )}
    </article>
  );
}

export default function Learning() {
  const [days, setDays] = useState(14);
  const [board, setBoard] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [previewQuery, setPreviewQuery] = useState("");
  const [previewHits, setPreviewHits] = useState([]);

  const load = useCallback(async () => {
    const res = await fetch(`/learning/summary?days=${days}`);
    if (!res.ok) throw new Error(await res.text());
    setBoard(await res.json());
  }, [days]);

  useEffect(() => {
    document.title = "Legal Model Learning";
    load().catch((err) => setError(err.message || "Failed to load learning"));
  }, [load]);

  const runAction = async (label, work) => {
    setBusy(label);
    setError("");
    try {
      await work();
      await load();
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setBusy("");
    }
  };

  const decide = (lessonId, action) =>
    runAction(action, async () => {
      const res = await fetch(`/learning/lessons/${lessonId}/${action}`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
    });

  const analyze = () =>
    runAction("analyze", async () => {
      const res = await fetch(`/learning/analyze?days=${days}`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
    });

  const retrieve = async (event) => {
    event.preventDefault();
    const q = previewQuery.trim();
    if (!q) return;
    setError("");
    const res = await fetch(`/learning/preview?q=${encodeURIComponent(q)}`);
    if (!res.ok) {
      setError(await res.text());
      return;
    }
    const body = await res.json();
    setPreviewHits(body.lessons || []);
  };

  const candidates = board?.candidates_items || [];
  const approved = board?.approved_items || [];
  const incidents = board?.incidents_items || [];

  return (
    <div className="learn-shell">
      <header className="learn-top">
        <div>
          <h1>Legal Model Learning</h1>
          <p>Logs + feedback → incidents → candidate lessons → approved runtime context.</p>
        </div>
        <div className="learn-controls">
          <select value={days} onChange={(e) => setDays(Number(e.target.value))} aria-label="Range">
            <option value={7}>7d</option>
            <option value={14}>14d</option>
            <option value={30}>30d</option>
          </select>
          <button type="button" className="primary" onClick={analyze} disabled={busy === "analyze"}>
            {busy === "analyze" ? "Analyzing…" : "Analyze logs"}
          </button>
          <a className="ghost-link" href="/learning/eval.jsonl">
            Eval JSONL
          </a>
          <a className="ghost-link" href="/observability">
            Observability
          </a>
          <a className="ghost-link" href="/">
            Query
          </a>
        </div>
      </header>
      {error && <p className="learn-error">{error}</p>}

      <section className="stat-row">
        {[
          ["Total Feedbacks", board?.total_feedbacks ?? 0],
          ["Approved feedback", board?.approved_feedback ?? 0],
          ["Incidents", board?.incidents ?? 0],
        ].map(([label, value]) => (
          <article key={label} className="stat-card">
            <p>{label}</p>
            <strong>{value}</strong>
          </article>
        ))}
      </section>

      <section className="lesson-columns">
        <div>
          <h2>Candidate lessons</h2>
          {candidates.length === 0 && <p className="empty">No candidate lessons.</p>}
          {candidates.map((lesson) => (
            <LessonCard
              key={lesson.lesson_id}
              lesson={lesson}
              onApprove={(id) => decide(id, "approve")}
              onReject={(id) => decide(id, "reject")}
            />
          ))}
        </div>
        <div>
          <h2>Approved lessons</h2>
          {approved.length === 0 && <p className="empty">No approved lessons.</p>}
          {approved.map((lesson) => (
            <LessonCard key={lesson.lesson_id} lesson={lesson} />
          ))}
        </div>
      </section>

      <section className="lower-grid">
        <div className="panel">
          <h2>Recent incidents</h2>
          {incidents.length === 0 ? (
            <p className="empty">No incidents.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Type</th>
                  <th>Area</th>
                  <th>Route</th>
                  <th>Query</th>
                  <th>Feedback/Error</th>
                </tr>
              </thead>
              <tbody>
                {incidents.map((row) => (
                  <tr key={row.incident_id}>
                    <td>{row.type}</td>
                    <td>{row.area}</td>
                    <td>{row.route}</td>
                    <td>{row.query}</td>
                    <td>{row.feedback}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <form className="preview" onSubmit={retrieve}>
            <h2>Learning context preview</h2>
            <div className="inline">
              <input
                value={previewQuery}
                onChange={(e) => setPreviewQuery(e.target.value)}
                placeholder="Query to test approved lesson retrieval"
              />
              <button type="submit" className="primary">
                Retrieve
              </button>
            </div>
            <div className="preview-box">
              {previewHits.length === 0 && <p className="empty">No approved lessons retrieved.</p>}
              {previewHits.map((hit) => (
                <article key={hit.lesson_id}>
                  <strong>{hit.title}</strong>
                  <p>{hit.lesson}</p>
                </article>
              ))}
            </div>
          </form>
        </div>
      </section>
    </div>
  );
}
