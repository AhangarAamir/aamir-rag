import { useCallback, useEffect, useMemo, useState } from "react";
import PlotChart from "./PlotChart.jsx";
import "./Observability.css";

const MINT = "#3ee0a8";
const GOLD = "#f0c14b";
const BLUE = "#6ea8ff";
const PINK = "#f472b6";
const GREEN_BAR = "#3dd6a7";

function fmtInt(n) {
  return new Intl.NumberFormat("en-US").format(Math.round(n || 0));
}

function fmtUsd(n) {
  return `$${(n || 0).toFixed(6)}`;
}

function fmtMs(ms) {
  if (ms == null || Number.isNaN(ms)) return "—";
  if (Math.abs(ms) < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

const LATENCY_AXIS = {
  title: "Latency (seconds)",
  rangemode: "nonnegative",
  tickformat: ",.0f",
  exponentformat: "none",
  separatethousands: true,
};

function latencyTickStep(maxSeconds) {
  const max = Math.max(0, maxSeconds);
  if (max <= 100) return 10;
  const target = Math.max(max / 8, 10);
  const pow = 10 ** Math.floor(Math.log10(target));
  const scaled = target / pow;
  const nice = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return nice * pow;
}

function shortDate(day) {
  const d = new Date(`${day}T00:00:00Z`);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

function localTime(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
}

export default function Observability() {
  const [days, setDays] = useState(14);
  const [app, setApp] = useState("");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [comment, setComment] = useState("");

  const load = useCallback(async () => {
    try {
      setError("");
      const params = new URLSearchParams({ days: String(days) });
      if (app) params.set("app", app);
      const res = await fetch(`/observability/summary?${params}`);
      if (!res.ok) throw new Error(await res.text());
      setData(await res.json());
    } catch (err) {
      setError(err.message || "Failed to load observability data");
    }
  }, [days, app]);

  useEffect(() => {
    document.title = "BIF Search Observability";
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const series = data?.days_series || [];
  const x = series.map((row) => shortDate(row.date));

  const runsChart = useMemo(
    () => [
      {
        type: "bar",
        name: "ok",
        x,
        y: series.map((row) => row.ok),
        marker: { color: GREEN_BAR },
      },
      {
        type: "bar",
        name: "fail",
        x,
        y: series.map((row) => row.fail),
        marker: { color: "#f59e0b" },
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "success %",
        yaxis: "y2",
        x,
        y: series.map((row) => {
          const total = row.ok + row.fail + row.run;
          return total ? (row.ok / total) * 100 : 0;
        }),
        line: { color: BLUE, width: 2 },
        marker: { size: 6 },
      },
    ],
    [series, x],
  );

  const mix = data?.query_mix || [];
  const mixChart = useMemo(
    () => [
      {
        type: "bar",
        orientation: "h",
        x: mix.map((row) => row.count),
        y: mix.map((row) => row.name),
        marker: { color: MINT },
        text: mix.map((row) => `${row.count} · ${row.success.toFixed(0)}%`),
        textposition: "outside",
        cliponaxis: false,
      },
    ],
    [mix],
  );

  const latencyChart = useMemo(
    () => ({
      data: [
        {
          type: "scatter",
          mode: "lines+markers",
          name: "avg",
          x,
          y: series.map((row) => (row.avg_ms || 0) / 1000),
          line: { color: BLUE, width: 2 },
          hovertemplate: "avg: %{y:,.1f} s<extra></extra>",
        },
        {
          type: "scatter",
          mode: "lines+markers",
          name: "p95",
          x,
          y: series.map((row) => (row.p95_ms || 0) / 1000),
          line: { color: GOLD, width: 2 },
          hovertemplate: "p95: %{y:,.1f} s<extra></extra>",
        },
      ],
      layout: { yaxis: { ...LATENCY_AXIS } },
    }),
    [series, x],
  );

  const routes = [...(data?.routes || [])].slice().reverse();
  const routeChart = useMemo(
    () => [
      {
        type: "bar",
        orientation: "h",
        x: routes.map((row) => row.count),
        y: routes.map((row) => row.name),
        marker: { color: MINT },
        text: routes.map(
          (row) => `${row.count} · ${row.success.toFixed(0)}% · ${fmtMs(row.avg_ms)}`,
        ),
        textposition: "outside",
        cliponaxis: false,
      },
    ],
    [routes],
  );

  const scatter = useMemo(() => {
    const points = data?.scatter || [];
    const seconds = points.map(
      (row) => Number(row.latency_s ?? (row.latency_ms || 0) / 1000) || 0,
    );
    const maxS = seconds.reduce((max, value) => Math.max(max, value), 0);
    const dtick = latencyTickStep(maxS);
    return {
      chart: [
        {
          type: "scatter",
          mode: "markers",
          name: "run",
          x: points.map((row) => row.tokens),
          y: seconds,
          marker: { color: BLUE, size: 9, opacity: 0.85 },
          text: points.map((row) => row.route),
          hovertemplate: "Tokens: %{x:,}<br>Latency: %{y:,.1f} s<br>%{text}<extra></extra>",
        },
      ],
      layout: {
        margin: { t: 36, r: 24, b: 56, l: maxS >= 1000 ? 84 : 64 },
        xaxis: { title: "Tokens", rangemode: "nonnegative" },
        yaxis: {
          ...LATENCY_AXIS,
          tickmode: "linear",
          tick0: 0,
          dtick,
        },
        showlegend: false,
      },
    };
  }, [data]);

  const pricingChart = useMemo(
    () => [
      {
        type: "bar",
        name: "input",
        x,
        y: series.map((row) => row.cost_input ?? 0),
        marker: { color: BLUE },
      },
      {
        type: "bar",
        name: "cached",
        x,
        y: series.map((row) => row.cost_cached ?? 0),
        marker: { color: MINT },
      },
      {
        type: "bar",
        name: "output",
        x,
        y: series.map((row) => row.cost_output ?? 0),
        marker: { color: GOLD },
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "total",
        x,
        y: series.map((row) => row.cost),
        line: { color: PINK, width: 2 },
      },
    ],
    [series, x],
  );

  const addComment = async (event) => {
    event.preventDefault();
    if (!comment.trim()) return;
    const res = await fetch("/observability/comments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comment: comment.trim() }),
    });
    if (!res.ok) {
      setError(await res.text());
      return;
    }
    setComment("");
    load();
  };

  const success = data ? data.success_rate.toFixed(0) : "0";

  return (
    <div className="obs-shell">
      <header className="obs-top">
        <div>
          <p className="obs-kicker">BIF Search Observability</p>
          <p className="obs-sub">
            {days}d : {data?.runs ?? 0} runs - {success}% success - {fmtUsd(data?.cost)} -{" "}
            {localTime(data?.generated_at)}
          </p>
        </div>
        <div className="obs-controls">
          <a href="/">Query UI</a>
          <select value={app} onChange={(e) => setApp(e.target.value)} aria-label="App">
            <option value="">App</option>
            {(data?.apps || ["legal-rag"]).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <select value={days} onChange={(e) => setDays(Number(e.target.value))} aria-label="Range">
            <option value={1}>1d</option>
            <option value={7}>7d</option>
            <option value={14}>14d</option>
            <option value={30}>30d</option>
            <option value={90}>90d</option>
          </select>
          <button type="button" className="primary" onClick={load}>
            Refresh
          </button>
        </div>
      </header>
      {error && <p className="obs-error">{error}</p>}

      <div className="obs-grid">
        <section className="kpi-row">
          <article className="obs-card">
            <div className="kpi-label">Runs</div>
            <p className="kpi-value">{data?.runs ?? 0}</p>
            <p className="kpi-meta">{data?.running ?? 0} running</p>
          </article>
          <article className="obs-card">
            <div className="kpi-label">Success</div>
            <p className="kpi-value">{success}%</p>
            <p className="kpi-meta">
              {data?.ok ?? 0} ok / {data?.fail ?? 0} fail
            </p>
          </article>
          <article className="obs-card">
            <div className="kpi-label">Avg</div>
            <p className="kpi-value">{fmtMs(data?.avg_ms)}</p>
            <p className="kpi-meta">median {fmtMs(data?.median_ms)}</p>
          </article>
          <article className="obs-card">
            <div className="kpi-label">P95</div>
            <p className="kpi-value">{fmtMs(data?.p95_ms)}</p>
            <p className="kpi-meta">tail latency</p>
          </article>
        </section>

        <section className="kpi-row second">
          <article className="obs-card">
            <div className="kpi-label">Tokens</div>
            <p className="kpi-value">{fmtInt(data?.tokens)}</p>
            <p className="kpi-meta">
              {fmtInt(data?.input_tokens)} in · {fmtInt(data?.output_tokens)} out
            </p>
          </article>
          <article className="obs-card">
            <div className="kpi-label">Cost</div>
            <p className="kpi-value">{fmtUsd(data?.cost)}</p>
            <p className="kpi-meta">estimated</p>
          </article>
          <article className="obs-card">
            <div className="kpi-label">Feedback</div>
            <p className="kpi-value">{data?.feedback ?? 0}</p>
            <p className="kpi-meta">{(data?.feedback_rate || 0).toFixed(0)}% rate</p>
          </article>
        </section>

        <section className="chip-row">
          <div className="obs-chip">
            <span className="dot green" />
            <div className="chip-copy">
              <small>Reliability</small>
              <strong>{(data?.success_rate || 0).toFixed(1)}% success</strong>
            </div>
          </div>
          <div className="obs-chip">
            <span className="dot blue" />
            <div className="chip-copy">
              <small>Peak day</small>
              <strong>
                {data?.peak_day || "—"}
                {data?.peak_day_runs ? `: ${data.peak_day_runs} runs` : ""}
              </strong>
            </div>
          </div>
          <div className="obs-chip">
            <span className="dot gold" />
            <div className="chip-copy">
              <small>Slowest class</small>
              <strong>
                {data?.slowest_class || "—"}
                {data?.slowest_class_ms ? ` · ${fmtMs(data.slowest_class_ms)} avg` : ""}
              </strong>
            </div>
          </div>
          <div className="obs-chip">
            <span className="dot orange" />
            <div className="chip-copy">
              <small>Slowest route</small>
              <strong>
                {data?.slowest_route || "—"}
                {data?.slowest_route_ms ? ` · ${fmtMs(data.slowest_route_ms)}` : ""}
              </strong>
            </div>
          </div>
          <div className="obs-chip">
            <span className="dot white" />
            <div className="chip-copy">
              <small>Model</small>
              <strong>{data?.model || "—"}</strong>
            </div>
          </div>
        </section>

        <section className="chart-row">
          <article className="obs-card">
            <div className="card-head">
              <h2>Runs & success</h2>
              <span className="kpi-meta">{data?.runs ?? 0} runs</span>
            </div>
            <PlotChart
              data={runsChart}
              layout={{
                barmode: "stack",
                yaxis: { title: "runs", gridcolor: "rgba(255,255,255,0.06)" },
                yaxis2: {
                  overlaying: "y",
                  side: "right",
                  title: "%",
                  range: [0, 100],
                  showgrid: false,
                },
              }}
            />
          </article>
          <article className="obs-card">
            <div className="card-head">
              <h2>Query mix</h2>
              <span className="kpi-meta">JOA · PEL · PML · PSC · RSC</span>
            </div>
            <PlotChart
              data={mixChart}
              layout={{
                margin: { t: 16, r: 80, b: 32, l: 120 },
                xaxis: { title: "", rangemode: "tozero" },
                showlegend: false,
              }}
            />
          </article>
        </section>

        <section className="chart-row">
          <article className="obs-card">
            <div className="card-head">
              <h2>Latency</h2>
              <span className="kpi-meta">p95 {fmtMs(data?.p95_ms)}</span>
            </div>
            <PlotChart data={latencyChart.data} layout={latencyChart.layout} />
          </article>
          <article className="obs-card">
            <div className="card-head">
              <h2>Routes</h2>
              <span className="kpi-meta">{routes.length} routes</span>
            </div>
            <PlotChart
              data={routeChart}
              layout={{
                margin: { t: 16, r: 110, b: 32, l: 140 },
                xaxis: { rangemode: "tozero" },
                showlegend: false,
              }}
            />
          </article>
        </section>

        <section className="chart-row">
          <article className="obs-card">
            <div className="card-head">
              <h2>Tokens vs latency</h2>
            </div>
            <PlotChart data={scatter.chart} layout={scatter.layout} />
          </article>
          <article className="obs-card">
            <div className="card-head">
              <h2>Day-wise model pricing</h2>
              <span className="kpi-meta">{fmtUsd(data?.cost)}</span>
            </div>
            <PlotChart data={pricingChart} layout={{ yaxis: { title: "USD" } }} />
            <table className="model-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Runs</th>
                  <th>Cost</th>
                  <th>Input</th>
                  <th>Cached</th>
                  <th>Output</th>
                </tr>
              </thead>
              <tbody>
                {[...series].reverse().map((row) => (
                  <tr key={row.date}>
                    <td>{row.date}</td>
                    <td>{row.ok + row.fail + row.run}</td>
                    <td>{fmtUsd(row.cost)}</td>
                    <td>{fmtInt(row.input_tokens)}</td>
                    <td>{fmtInt(row.cached_tokens)}</td>
                    <td>{fmtInt(row.output_tokens)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </article>
        </section>

        <section className="chart-row">
          <article className="obs-card">
            <div className="card-head">
              <h2>Model & pricing</h2>
              <span className="kpi-meta">{data?.model || "—"}</span>
            </div>
            {(data?.models || []).map((model) => (
              <div key={model.name} style={{ marginTop: "0.7rem" }}>
                <table className="model-table">
                  <tbody>
                    <tr>
                      <td>Pricing</td>
                      <td>
                        Input ${model.input_rate} · cached ${model.cached_rate} · output $
                        {model.output_rate} / 1M
                      </td>
                    </tr>
                    <tr>
                      <td>Rate source</td>
                      <td>
                        {model.source || data?.pricing?.source || "—"}
                        {data?.pricing?.fetched_at
                          ? ` · ${new Date(data.pricing.fetched_at).toLocaleString()}`
                          : ""}
                      </td>
                    </tr>
                    <tr>
                      <td>Estimated cost</td>
                      <td>{fmtUsd(model.cost)}</td>
                    </tr>
                    <tr>
                      <td>{model.name}</td>
                      <td>
                        {model.runs} runs · {fmtInt(model.input_tokens)} in ·{" "}
                        {fmtInt(model.output_tokens)} out
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ))}
            {!data?.models?.length && <p className="comment-empty">No data</p>}
          </article>
          <article className="obs-card comments">
            <div className="card-head">
              <h2>User comments</h2>
              <span className="kpi-meta">{data?.feedback ?? 0} shown</span>
            </div>
            {(data?.comments || []).length === 0 ? (
              <p className="comment-empty">No data</p>
            ) : (
              <table className="model-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Comment</th>
                    <th>Query</th>
                    <th>History</th>
                  </tr>
                </thead>
                <tbody>
                  {data.comments.map((row) => (
                    <tr key={row.comment_id}>
                      <td>{localTime(row.created_at)}</td>
                      <td>{row.comment}</td>
                      <td>{row.query || "—"}</td>
                      <td>{row.history || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <form className="comment-form" onSubmit={addComment}>
              <input
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                placeholder="Add a comment"
              />
              <button type="submit">Save</button>
            </form>
          </article>
        </section>
      </div>
    </div>
  );
}
