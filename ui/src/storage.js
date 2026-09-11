const USER_KEY = "legal-rag-user-id";

export function getUserId() {
  let id = localStorage.getItem(USER_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(USER_KEY, id);
  }
  return id;
}

export function newSessionId() {
  return crypto.randomUUID();
}
