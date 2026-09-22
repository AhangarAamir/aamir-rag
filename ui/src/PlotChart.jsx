import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

const BASE_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#9fb4cc", family: "ui-sans-serif, system-ui, sans-serif", size: 11 },
  margin: { t: 36, r: 18, b: 40, l: 48 },
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

function axisTitle(title) {
  if (title == null || title === "") return title;
  if (typeof title === "string") {
    return { text: title, font: { color: "#d7e6f7", size: 12 }, standoff: 8 };
  }
  return { font: { color: "#d7e6f7", size: 12 }, standoff: 8, ...title };
}

function withAxis(base, extra) {
  const axis = { ...base, ...(extra || {}) };
  if (axis.title != null && axis.title !== "") {
    axis.title = axisTitle(axis.title);
  }
  return axis;
}

function mergedLayout(layout, height) {
  const next = {
    ...BASE_LAYOUT,
    height,
    dragmode: "zoom",
    ...layout,
    xaxis: withAxis(BASE_LAYOUT.xaxis, layout.xaxis),
    yaxis: withAxis(BASE_LAYOUT.yaxis, layout.yaxis),
  };
  if (layout.yaxis2) {
    next.yaxis2 = withAxis(
      {
        zeroline: false,
        linecolor: "rgba(255,255,255,0.08)",
        tickfont: { color: "#7f96b0" },
      },
      layout.yaxis2,
    );
  }
  return next;
}

export default function PlotChart({ data, layout = {}, modeBar = true, height = 240 }) {
  const el = useRef(null);
  const layoutKey = JSON.stringify(layout);

  useEffect(() => {
    if (!el.current) return;
    Plotly.react(el.current, data, mergedLayout(layout, height), {
      displayModeBar: modeBar,
      displaylogo: false,
      responsive: true,
      scrollZoom: true,
      modeBarButtonsToRemove: ["sendDataToCloud"],
    });
  }, [data, layout, layoutKey, modeBar, height]);

  useEffect(() => {
    const node = el.current;
    const onResize = () => {
      if (node) Plotly.Plots.resize(node);
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      if (node) Plotly.purge(node);
    };
  }, []);

  return <div ref={el} className="plot-host" />;
}
