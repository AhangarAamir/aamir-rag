export async function parseSse(response, onEvent) {
  if (!response.body) {
    throw new Error("No response body to stream");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      if (!part.trim() || part.startsWith(":")) continue;
      let event = "message";
      const dataLines = [];
      for (const line of part.split("\n")) {
        if (line.startsWith("event:")) {
          event = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
        }
      }
      const raw = dataLines.join("\n");
      if (!raw) continue;
      let payload = raw;
      try {
        payload = JSON.parse(raw);
      } catch {
        payload = { content: raw };
      }
      onEvent(event, payload);
    }
  }
}
