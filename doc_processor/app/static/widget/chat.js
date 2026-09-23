
const API_BASE = "http://localhost:9000";
let sessionId = null;
let selectedRating = 0;
let cachedPersonas = [];
let hasPersonaChoice = false;

function getTenantId() {
  const params = new URLSearchParams(window.location.search);
  return params.get("tenant");
}

function getEmbedOrigin() {
  const params = new URLSearchParams(window.location.search);
  return params.get("embed_origin");
}

function getSessionStorageKey() {
  const tenantId = getTenantId();
  return `liquidlab_persona_sessions_${tenantId || "default"}`;
}

function loadSessionMap() {
  try {
    return JSON.parse(localStorage.getItem(getSessionStorageKey())) || {};
  } catch (e) {
    return {};
  }
}

function saveSessionForPersona(slug, id) {
  const key = slug || "__default__";
  const map = loadSessionMap();
  map[key] = id;
  try {
    localStorage.setItem(getSessionStorageKey(), JSON.stringify(map));
  } catch (e) { /* storage disabled/full — degrade to always-fresh, no crash */ }
}

function getSessionForPersona(slug) {
  const key = slug || "__default__";
  return loadSessionMap()[key] || null;
}

function notifyParentSessionActive(id) {
  if (window.parent !== window) {
    window.parent.postMessage({ type: "liquidlab-session-created", sessionId: id }, "*");
  }
}

async function fetchAndRenderMessages(id) {
  const res = await fetch(`${API_BASE}/chats/${id}/messages`);
  if (!res.ok) return false;
  const messages = await res.json();
  messages.forEach((m) => renderMessage(m.role, m.content));
  return true;
}

let companyName = null;

async function loadCompanyName() {
  const tenantId = getTenantId();
  if (!tenantId) return;
  try {
    const res = await fetch(`${API_BASE}/widget/company`, {
      headers: { "X-API-Key": tenantId, "X-Embed-Origin": getEmbedOrigin() || "" },
    });
    if (!res.ok) return;
    const { name } = await res.json();
    companyName = name;
    document.getElementById("assistant-title").textContent = `${name} Assistant`;
    document.getElementById("query-input").placeholder = `Ask about ${name}...`;
    document.title = `${name} Chat`;
  } catch (e) { /* keep generic labels */ }
}

async function fetchPersonas() {
  const tenantId = getTenantId();
  if (!tenantId) return [];
  try {
    const res = await fetch(`${API_BASE}/widget/personas`, {
      headers: { "X-API-Key": tenantId, "X-Embed-Origin": getEmbedOrigin() || "" },
    });
    if (!res.ok) return [];
    return await res.json();
  } catch (e) {
    return [];
  }
}

function showPersonaPicker() {
  document.getElementById("messages").classList.add("hidden");
  document.getElementById("input-row").classList.add("hidden");
  document.getElementById("back-btn").classList.add("hidden");

  if (companyName) {
    document.getElementById("assistant-title").textContent = `${companyName} Assistant`;
  }

  const optionsDiv = document.getElementById("persona-options");
  optionsDiv.innerHTML = "";
  cachedPersonas.forEach((p) => {
    const btn = document.createElement("button");
    btn.className = "persona-option";
    btn.innerHTML = `<span class="persona-option-name">${p.name}</span>` +
      (p.description ? `<span class="persona-option-desc">${p.description}</span>` : "");
    btn.addEventListener("click", () => selectPersona(p.slug, p.name));
    optionsDiv.appendChild(btn);
  });

  document.getElementById("persona-picker").classList.remove("hidden");
}

function hidePersonaPicker() {
  document.getElementById("persona-picker").classList.add("hidden");
  document.getElementById("messages").classList.remove("hidden");
  document.getElementById("input-row").classList.remove("hidden");
  if (hasPersonaChoice) document.getElementById("back-btn").classList.remove("hidden");
}

async function selectPersona(slug, name) {
  document.getElementById("messages").innerHTML = "";
  if (name) document.getElementById("assistant-title").textContent = name;
  hidePersonaPicker();

  const existingId = getSessionForPersona(slug);
  if (existingId) {
    sessionId = existingId;
    const resumed = await fetchAndRenderMessages(existingId);
    if (resumed) {
      notifyParentSessionActive(existingId);
      return;
    }
    // session was deleted/expired server-side — fall through to create fresh
  }

  await createSession(slug);
}

function formatText(text) {
  let safe = text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  const lines = safe.split("\n");
  let html = "";
  let inList = false;
  for (const line of lines) {
    const bulletMatch = line.match(/^-\s+(.*)/);
    if (bulletMatch) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${bulletMatch[1]}</li>`;
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      if (line.trim() !== "") html += `<p>${line}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function renderMessage(role, text) {
  const messagesDiv = document.getElementById("messages");
  const msgEl = document.createElement("div");
  msgEl.className = "message " + role;
  msgEl.innerHTML = formatText(text);
  messagesDiv.appendChild(msgEl);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function showTypingIndicator() {
  const messagesDiv = document.getElementById("messages");
  const bubble = document.createElement("div");
  bubble.id = "typing-indicator";
  bubble.className = "thinking-bubble";
  bubble.innerHTML = "<span></span><span></span><span></span>";
  messagesDiv.appendChild(bubble);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function hideTypingIndicator() {
  const bubble = document.getElementById("typing-indicator");
  if (bubble) bubble.remove();
}

async function createSession(personaSlug) {
  const tenantId = getTenantId();
  const embedOrigin = getEmbedOrigin();

  if (!tenantId) {
    renderMessage("assistant", "This chat widget isn't configured correctly. Please contact the site owner.");
    return;
  }

  const res = await fetch(`${API_BASE}/widget/session`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": tenantId,
      "X-Embed-Origin": embedOrigin || "",
    },
    body: JSON.stringify(personaSlug ? { persona_slug: personaSlug } : {}),
  });



  if (!res.ok) {
    renderMessage("assistant", "Sorry, I couldn't start a new conversation. Please try again shortly.");
    return;
  }

  const data = await res.json();
  sessionId = data.id;

  saveSessionForPersona(personaSlug, sessionId);

  const msgRes = await fetch(`${API_BASE}/chats/${sessionId}/messages`);
  if (msgRes.ok) (await msgRes.json()).forEach((m) => renderMessage(m.role, m.content));

  if (window.parent !== window) {
    window.parent.postMessage({ type: "liquidlab-session-created", sessionId: sessionId }, "*");
  }
}

async function loadExistingSession(existingId) {
  sessionId = existingId;

  cachedPersonas = await fetchPersonas();
  hasPersonaChoice = cachedPersonas.length > 1;

  const tenantId = getTenantId();
  const embedOrigin = getEmbedOrigin();
  let personaSlug = null;
  try {
    const sessRes = await fetch(`${API_BASE}/widget/session/${sessionId}`, {
      headers: { "X-API-Key": tenantId, "X-Embed-Origin": embedOrigin || "" },
    });
    if (sessRes.ok) {
      const sessData = await sessRes.json();
      personaSlug = sessData.persona_slug;
      if (sessData.persona_name) {
        document.getElementById("assistant-title").textContent = sessData.persona_name;
      }
    }
  } catch (e) { /* keep generic title */ }

  if (hasPersonaChoice) document.getElementById("back-btn").classList.remove("hidden");

  saveSessionForPersona(personaSlug, sessionId);

  const ok = await fetchAndRenderMessages(sessionId);
  if (!ok) {
    await createSession();
  }
}

async function initSession() {
  const params = new URLSearchParams(window.location.search);
  const existingId = params.get("sid");
  if (existingId) {
    loadExistingSession(existingId);
    return;
  }

  cachedPersonas = await fetchPersonas();

  if (cachedPersonas.length === 0) {
    createSession();
  } else if (cachedPersonas.length === 1) {
    hasPersonaChoice = false;
    selectPersona(cachedPersonas[0].slug, cachedPersonas[0].name);
  } else {
    hasPersonaChoice = true;
    showPersonaPicker();
  }
}

async function sendMessage() {
  const input = document.getElementById("query-input");
  const query = input.value.trim();
  if (!query || !sessionId) return;

  renderMessage("user", query);
  input.value = "";
  showTypingIndicator();

  const embedOrigin = getEmbedOrigin();

  const res = await fetch(`${API_BASE}/widget/chat?session_id=${sessionId}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Embed-Origin": embedOrigin || "",
    },
    body: JSON.stringify({ session_id: sessionId, query: query }),
  });

  hideTypingIndicator();

  if (!res.ok) {
    renderMessage("assistant", "Sorry, something went wrong. Please try again.");
    return;
  }

  const data = await res.json();
  renderMessage("assistant", data.answer);
}

function openFeedback() {
  document.getElementById("feedback-overlay").classList.remove("hidden");
}

function endConversation() {
  document.getElementById("feedback-overlay").classList.add("hidden");
  document.getElementById("ended-message").classList.remove("hidden");

  // Auto-close the widget a couple seconds after showing the thank-you
  // message, so the visitor doesn't have to click anything. Reopening via
  // the bubble (widget.js) will reset this view back to the conversation.
  if (window.parent !== window) {
    setTimeout(() => {
      window.parent.postMessage({ type: "liquidlab-close-widget" }, "*");
    }, 2000);
  }
}

async function submitFeedback() {
  if (sessionId && selectedRating > 0) {
    const comment = document.getElementById("feedback-comment").value.trim();
    await fetch(`${API_BASE}/chats/${sessionId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating: selectedRating, comment: comment || null }),
    });
  }
  endConversation();
}

document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("query-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});


document.getElementById("back-btn").addEventListener("click", () => {
  sessionId = null;
  document.getElementById("messages").innerHTML = "";
  showPersonaPicker();
});

document.getElementById("close-btn").addEventListener("click", openFeedback);
document.getElementById("skip-btn").addEventListener("click", endConversation);
document.getElementById("submit-feedback-btn").addEventListener("click", submitFeedback);

document.querySelectorAll(".star").forEach((star) => {
  star.addEventListener("click", () => {
    selectedRating = parseInt(star.dataset.value, 10);
    document.querySelectorAll(".star").forEach((s) => {
      s.classList.toggle("selected", parseInt(s.dataset.value, 10) <= selectedRating);
    });
  });
});

window.addEventListener("message", (event) => {
  if (event.data && event.data.type === "liquidlab-resume-chat") {
    document.getElementById("ended-message").classList.add("hidden");
    document.getElementById("messages").classList.remove("hidden");
    document.getElementById("input-row").classList.remove("hidden");
  }
});

(async () => {
  await loadCompanyName();
  await initSession();
})();