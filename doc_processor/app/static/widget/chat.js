const API_BASE = "http://localhost:9000";
let sessionId = null;
let selectedRating = 0;

function formatText(text) {
  return text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/^- (.+)$/gm, "• $1")
    .replace(/\n/g, "<br>");
}

function renderMessage(role, text) {
  const messagesDiv = document.getElementById("messages");
  const msgEl = document.createElement("div");
  msgEl.className = "message " + role;
  msgEl.innerHTML = formatText(text);
  messagesDiv.appendChild(msgEl);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function createSession() {
  const res = await fetch(`${API_BASE}/widget/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
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

  const res = await fetch(`${API_BASE}/widget/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, query: query }),
  });

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

initSession();