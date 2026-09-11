const API_BASE = "http://localhost:9000";
let sessionId = null;
let selectedRating = 0;

function getApiKey() {
  const params = new URLSearchParams(window.location.search);
  return params.get("key");
}

function getEmbedOrigin() {
  const params = new URLSearchParams(window.location.search);
  return params.get("embed_origin");
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

async function createSession() {
  const apiKey = getApiKey();
  const embedOrigin = getEmbedOrigin();

  if (!apiKey) {
    renderMessage("assistant", "This chat widget isn't configured correctly. Please contact the site owner.");
    return;
  }

  const res = await fetch(`${API_BASE}/widget/session`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": apiKey,
      "X-Embed-Origin": embedOrigin || "",
    },
    body: JSON.stringify({}),
  });

  if (!res.ok) {
    renderMessage("assistant", "Sorry, I couldn't start a new conversation. Please try again shortly.");
    return;
  }

  const data = await res.json();
  sessionId = data.id;
  renderMessage("assistant", "Hi! I'm the LiquidLab Assistant. Ask me anything about our services, solutions, or company.");

  if (window.parent !== window) {
    window.parent.postMessage({ type: "liquidlab-session-created", sessionId: sessionId }, "*");
  }
}

async function loadExistingSession(existingId) {
  sessionId = existingId;
  const res = await fetch(`${API_BASE}/chats/${sessionId}/messages`);
  if (!res.ok) {
    await createSession();
    return;
  }
  const messages = await res.json();
  messages.forEach((m) => renderMessage(m.role, m.content));
}

function initSession() {
  const params = new URLSearchParams(window.location.search);
  const existingId = params.get("sid");
  if (existingId) {
    loadExistingSession(existingId);
  } else {
    createSession();
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
  console.log("[chat.js] scheduling auto-close, window.parent !== window:", window.parent !== window);
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

// Parent (widget.js) tells us to reset back to the normal chat view when
// the visitor reopens the widget after a previous conversation ended.
window.addEventListener("message", (event) => {
  if (event.data && event.data.type === "liquidlab-resume-chat") {
    document.getElementById("ended-message").classList.add("hidden");
    document.getElementById("messages").classList.remove("hidden");
    document.getElementById("input-row").classList.remove("hidden");
  }
});

initSession();