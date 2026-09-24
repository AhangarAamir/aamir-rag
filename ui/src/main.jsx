import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import Learning from "./Learning.jsx";
import Observability from "./Observability.jsx";
import "./App.css";

const path = window.location.pathname;
const page = path.startsWith("/observability") ? (
  <Observability />
) : path.startsWith("/learning") ? (
  <Learning />
) : (
  <App />
);

createRoot(document.getElementById("root")).render(
  <React.StrictMode>{page}</React.StrictMode>,
);
