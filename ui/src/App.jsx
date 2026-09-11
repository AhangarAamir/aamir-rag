import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getSessionRuns, listSessions, runQuery, runsToMessages } from "./api.js";
import { getUserId, newSessionId } from "./storage.js";

function toolLabel(name) {
  if (name === "think") return "Thinking";
  if (name === "search_knowledge") return "Searching knowledge";
  if (name === "analyze") return "Analyzing";
  return name || "Tool";
}

function toolDetail(tool) {
  const args = tool.args || {};
  return args.thought || args.query || args.analysis || args.input || "";
}

function Activity({ tools }) {
  if (!tools?.length) return null;
  return (
    <div className="activity">
      {tools.map((tool) => (
        <details key={tool.id} className={`activity-item ${tool.status}`} open={tool.status === "running"}>
          <summary>
            <span className="dot" />
            {toolLabel(tool.name)}
            {tool.status === "running" ? "…" : ""}
          </summary>
          {toolDetail(tool) && <pre>{toolDetail(tool)}</pre>}
          {tool.status === "error" && tool.result && <pre className="error">{tool.result}</pre>}
        </details>
      ))}
    </div>
  );
}

function normalizeMarkdown(text) {
  if (!text) return text;
  return text.replace(/((?:^|\n)(?:\|[^\n]+)+)/g, (block) => {
    if (!/\|[\s:-]+\|/.test(block)) return block;
    if (block.includes("\n|")) return block;
    return block.replace(/\|\s+\|/g, "|\n|");
  });
}

function MessageBubble({ message }) {
  return (
    <article className={`bubble ${message.role}`}>
      <div className="bubble-role">{message.role === "user" ? "You" : "Query Agent"}</div>
      {message.role === "assistant" && <Activity tools={message.tools} />}
      {message.role === "user" ? (
        <>
          {message.files?.length > 0 && (
            <ul className="file-chips">
              {message.files.map((name) => (
                <li key={name}>{name}</li>
              ))}
            </ul>
          )}
          {message.content && <div className="bubble-text">{message.content}</div>}
        </>
      ) : (
        <div className="bubble-markdown">
          {message.content ? (
            <Markdown remarkPlugins={[remarkGfm]}>{normalizeMarkdown(message.content)}</Markdown>
          ) : message.streaming ? (
            <span className="cursor">▍</span>
          ) : null}
        </div>
      )}
      {message.error && <p className="bubble-error">{message.error}</p>}
    </article>
  );
}

export default function App() {
  const userId = useMemo(() => getUserId(), []);
  const [sessionId, setSessionId] = useState(() => newSessionId());
  const [sessions, setSessions] = useState([]);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [files, setFiles] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const scroller = useRef(null);
  const assistantRef = useRef(null);
  const fileInput = useRef(null);

  const refreshSessions = useCallback(async () => {
    try {
      const page = await listSessions(userId);
      setSessions(page.data || []);
    } catch (err) {
      setError(err.message);
    }
  }, [userId]);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const startNewChat = () => {
    if (streaming) return;
    setSessionId(newSessionId());
    setMessages([]);
    setFiles([]);
    setError("");
  };

  const openSession = async (id) => {
    if (streaming || id === sessionId) return;
    setSessionId(id);
    setError("");
    try {
      const runs = await getSessionRuns(id, userId);
      setMessages(runsToMessages(runs));
    } catch (err) {
      setMessages([]);
      setError(err.message);
    }
  };

  const addFiles = (list) => {
    const next = Array.from(list || []);
    if (!next.length) return;
    setFiles((prev) => {
      const seen = new Set(prev.map((f) => `${f.name}:${f.size}:${f.lastModified}`));
      const extra = next.filter((f) => !seen.has(`${f.name}:${f.size}:${f.lastModified}`));
      return extra.length ? [...prev, ...extra] : prev;
    });
  };

  const send = async (event) => {
    event?.preventDefault();
    const text = draft.trim();
    if ((!text && files.length === 0) || streaming) return;

    const attached = files;
    const messageText = text || (attached.length ? "Please review the attached file(s)." : "");
    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      files: attached.map((f) => f.name),
    };
    const assistantId = crypto.randomUUID();
    assistantRef.current = {
      id: assistantId,
      role: "assistant",
      content: "",
      tools: [],
      streaming: true,
    };
    setMessages((prev) => [...prev, userMessage, assistantRef.current]);
    setDraft("");
    setFiles([]);
    if (fileInput.current) fileInput.current.value = "";
    setStreaming(true);
    setError("");

    const patchAssistant = (updater) => {
      assistantRef.current = updater({ ...assistantRef.current });
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? assistantRef.current : m)));
    };

    try {
      await runQuery({
        message: messageText,
        sessionId,
        userId,
        files: attached,
        onEvent: (event, data) => {
          if (event === "ToolCallStarted") {
            const tool = data.tool || {};
            patchAssistant((msg) => ({
              ...msg,
              tools: [
                ...msg.tools,
                {
                  id: tool.tool_call_id || `${tool.tool_name}-${msg.tools.length}`,
                  name: tool.tool_name || "tool",
                  args: tool.tool_args || {},
                  result: "",
                  status: "running",
                },
              ],
            }));
            return;
          }
          if (event === "ToolCallCompleted" || event === "ToolCallError") {
            const tool = data.tool || {};
            patchAssistant((msg) => ({
              ...msg,
              tools: msg.tools.map((item) =>
                item.id === tool.tool_call_id || (item.name === tool.tool_name && item.status === "running")
                  ? {
                      ...item,
                      args: tool.tool_args || item.args,
                      result: tool.result || data.error || "",
                      status: event === "ToolCallError" || tool.tool_call_error ? "error" : "done",
                    }
                  : item,
              ),
            }));
            return;
          }
          if (event === "ReasoningStep" || event === "ReasoningContentDelta") {
            const thought = data.reasoning_content || data.content || "";
            if (!thought) return;
            patchAssistant((msg) => {
              const last = msg.tools[msg.tools.length - 1];
              if (last?.name === "reasoning" && last.status === "running") {
                return {
                  ...msg,
                  tools: msg.tools.map((item, i) =>
                    i === msg.tools.length - 1 ? { ...item, args: { thought: (item.args.thought || "") + thought } } : item,
                  ),
                };
              }
              return {
                ...msg,
                tools: [
                  ...msg.tools,
                  {
                    id: `reasoning-${msg.tools.length}`,
                    name: "reasoning",
                    args: { thought },
                    result: "",
                    status: "running",
                  },
                ],
              };
            });
            return;
          }
          if (event === "ReasoningCompleted") {
            patchAssistant((msg) => ({
              ...msg,
              tools: msg.tools.map((item) => (item.name === "reasoning" ? { ...item, status: "done" } : item)),
            }));
            return;
          }
          if (event === "RunContent") {
            const chunk = typeof data.content === "string" ? data.content : "";
            if (!chunk) return;
            patchAssistant((msg) => ({ ...msg, content: msg.content + chunk }));
            return;
          }
          if (event === "RunError") {
            patchAssistant((msg) => ({ ...msg, error: data.content || "Run failed", streaming: false }));
            return;
          }
          if (event === "RunCompleted" || event === "RunCancelled") {
            patchAssistant((msg) => ({ ...msg, streaming: false }));
          }
        },
      });
      patchAssistant((msg) => ({ ...msg, streaming: false }));
      await refreshSessions();
    } catch (err) {
      patchAssistant((msg) => ({ ...msg, error: err.message, streaming: false }));
      setError(err.message);
    } finally {
      setStreaming(false);
    }
  };

  const onKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  };

  return (
    <div className={`app ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <aside className="sidebar">
        <div className="sidebar-head">
          <div>
            <p className="eyebrow">Legal RAG</p>
            <h1>Query Agent</h1>
          </div>
          <button type="button" className="ghost" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar">
            ✕
          </button>
        </div>
        <button type="button" className="new-chat" onClick={startNewChat} disabled={streaming}>
          New chat
        </button>
        <p className="session-label">Previous sessions</p>
        <nav className="session-list">
          {sessions.length === 0 && <p className="empty">No chats for this browser yet.</p>}
          {sessions.map((session) => (
            <button
              key={session.session_id}
              type="button"
              className={session.session_id === sessionId ? "session active" : "session"}
              onClick={() => openSession(session.session_id)}
              disabled={streaming}
            >
              <span>{session.session_name || "Untitled chat"}</span>
              {session.updated_at && <small>{new Date(session.updated_at).toLocaleString()}</small>}
            </button>
          ))}
        </nav>
        <p className="user-chip" title={userId}>
          Browser user · {userId.slice(0, 8)}
        </p>
      </aside>

      <main className="main">
        {!sidebarOpen && (
          <button type="button" className="open-sidebar" onClick={() => setSidebarOpen(true)}>
            Sessions
          </button>
        )}
        <div className="thread" ref={scroller}>
          {messages.length === 0 && (
            <div className="welcome">
              <h2>Ask a contract question</h2>
              <p>
                Answers come only from ingested instruments. The agent will think, search, and analyze before it
                streams a cited response.
              </p>
            </div>
          )}
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </div>
        {error && <p className="banner-error">{error}</p>}
        <form className="composer" onSubmit={send}>
          {files.length > 0 && (
            <ul className="file-chips pending">
              {files.map((file) => (
                <li key={`${file.name}:${file.size}:${file.lastModified}`}>
                  {file.name}
                  <button
                    type="button"
                    className="chip-remove"
                    onClick={() => setFiles((prev) => prev.filter((f) => f !== file))}
                    aria-label={`Remove ${file.name}`}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
          <div className="composer-row">
            <input
              ref={fileInput}
              type="file"
              multiple
              hidden
              accept=".pdf,.doc,.docx,.txt,.md,.csv,.json,.png,.jpg,.jpeg,.webp,.xlsx,.xls,.pptx,.ppt"
              onChange={(e) => {
                addFiles(e.target.files);
                e.target.value = "";
              }}
            />
            <button
              type="button"
              className="attach"
              onClick={() => fileInput.current?.click()}
              disabled={streaming}
            >
              Attach
            </button>
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask a question, or attach a file…"
              rows={2}
              disabled={streaming}
            />
            <button type="submit" disabled={streaming || (!draft.trim() && files.length === 0)}>
              {streaming ? "Running" : "Send"}
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
