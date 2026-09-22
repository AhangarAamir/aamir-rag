import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

const BASE_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#9fb4cc", family: "ui-sans-serif, system-ui, sans-serif", size: 11 },
  margin: { t: 28, r: 18, b: 40, l: 48 },
  xaxis: {
    gridcolor: "rgba(255,255,255,0.06)",
    zeroline: false,
    linecolor: "rgba(255,255,255,0.08)",
    tickfont: { color: "#7f96b0" },
  },
  yaxis: {
    gridcolor: "rgba(255,255,255,0.06)",
    zeroline: false,
    linecolor: "rgba(255,255,255,0.08)",
    tickfont: { color: "#7f96b0" },
  },
  legend: { orientation: "h", y: -0.22, font: { size: 11, color: "#9fb4cc" } },
  hoverlabel: { bgcolor: "#12233a", bordercolor: "#1e3a5a", font: { color: "#e8f1fb" } },
};

export default function PlotChart({ data, layout, modeBar = false, height = 240 }) {
  const el = useRef(null);

  useEffect(() => {
    if (!el.current) return;
    Plotly.react(
      el.current,
      data,
      { ...BASE_LAYOUT, height, ...layout },
      {
        displayModeBar: modeBar,
        displaylogo: false,
        responsive: true,
        modeBarButtonsToRemove: ["sendDataToCloud", "lasso2d", "select2d"],
      },
    );
    const onResize = () => Plotly.Plots.resize(el.current);
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      if (el.current) Plotly.purge(el.current);
    };
  }, [data, layout, modeBar, height]);

  return <div ref={el} className="plot-host" />;
}
