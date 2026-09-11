import { parseSse } from "./sse.js";

export const AGENT_ID = "query-agent";
export const DB_ID = "query-sessions-db";

async function readError(res) {
  try {
    const body = await res.json();
    return body.detail || JSON.stringify(body);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

export async function listSessions(userId) {
  const params = new URLSearchParams({
    type: "agent",
    component_id: AGENT_ID,
    user_id: userId,
    db_id: DB_ID,
    sort_by: "updated_at",
    sort_order: "desc",
    limit: "50",
  });
  const res = await fetch(`/sessions?${params}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function getSessionRuns(sessionId, userId) {
  const params = new URLSearchParams({
    type: "agent",
    user_id: userId,
    db_id: DB_ID,
  });
  const res = await fetch(`/sessions/${encodeURIComponent(sessionId)}/runs?${params}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function runQuery({ message, sessionId, userId, files = [], onEvent }) {
  const body = new FormData();
  body.append("message", message);
  body.append("stream", "true");
  body.append("session_id", sessionId);
  body.append("user_id", userId);
  for (const file of files) {
    body.append("files", file);
  }
  const res = await fetch(`/agents/${AGENT_ID}/runs`, {
    method: "POST",
    body,
  });
  if (!res.ok) throw new Error(await readError(res));
  await parseSse(res, onEvent);
}

export function runsToMessages(runs) {
  const messages = [];
  for (const run of runs || []) {
    const input = run.run_input || "";
    if (input) {
      messages.push({
        id: `${run.run_id}-user`,
        role: "user",
        content: input,
        files: fileNamesFromRun(run),
      });
    }
    messages.push({
      id: `${run.run_id}-assistant`,
      role: "assistant",
      content: typeof run.content === "string" ? run.content : run.content ? JSON.stringify(run.content) : "",
      tools: (run.tools || []).map((tool) => ({
        id: tool.tool_call_id || tool.tool_name,
        name: tool.tool_name || "tool",
        args: tool.tool_args || {},
        result: tool.result || "",
        status: tool.tool_call_error ? "error" : "done",
      })),
    });
  }
  return messages;
}

function fileNamesFromRun(run) {
  const names = [];
  for (const item of run.files || []) {
    if (item?.filename || item?.name) names.push(item.filename || item.name);
  }
  const media = run.input_media || {};
  for (const key of ["files", "images", "audio", "videos"]) {
    for (const item of media[key] || []) {
      const name = item?.filename || item?.name;
      if (name) names.push(name);
    }
  }
  return [...new Set(names)];
}
