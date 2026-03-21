const TOKEN = "pt_token";
const ROLE = "pt_role";
const USER = "pt_username";

export function getToken() {
  return localStorage.getItem(TOKEN);
}

export function authHeaders(json = false) {
  const t = getToken();
  const h = {};
  if (t) h.Authorization = `Bearer ${t}`;
  if (json) h["Content-Type"] = "application/json";
  return h;
}

export function setSession({ access_token, role, username }) {
  localStorage.setItem(TOKEN, access_token);
  localStorage.setItem(ROLE, role);
  localStorage.setItem(USER, username);
}

export function clearSession() {
  localStorage.removeItem(TOKEN);
  localStorage.removeItem(ROLE);
  localStorage.removeItem(USER);
}

export function getRole() {
  return localStorage.getItem(ROLE) || "";
}

export function getUsername() {
  return localStorage.getItem(USER) || "";
}
