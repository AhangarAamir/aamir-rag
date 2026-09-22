import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import Observability from "./Observability.jsx";
import "./App.css";

const page = window.location.pathname.startsWith("/observability") ? (
  <Observability />
) : (
  <App />
);

createRoot(document.getElementById("root")).render(
  <React.StrictMode>{page}</React.StrictMode>,
);
