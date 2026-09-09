(function () {
  const scriptTag = document.currentScript;
  const apiBase = scriptTag.getAttribute("data-api");
  const STORAGE_KEY = "liquidlab_chat_session_id";

  // --- Floating bubble button ---
  const bubble = document.createElement("button");
  bubble.textContent = "💬";
  bubble.style.cssText = `
    position: fixed; bottom: 20px; right: 20px;
    width: 56px; height: 56px; border-radius: 50%;
    background: #333; color: white; border: none;
    font-size: 24px; cursor: pointer; z-index: 999999;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  `;

  // --- Iframe (hidden until opened) ---
  const iframe = document.createElement("iframe");
  iframe.style.cssText = `
    position: fixed; bottom: 90px; right: 20px;
    width: 360px; height: 520px; border: none;
    border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    z-index: 999999; display: none;
  `;

  function buildIframeSrc() {
    const existingId = localStorage.getItem(STORAGE_KEY);
    const url = new URL(`${apiBase}/widget-ui/chat.html`);
    if (existingId) {
      url.searchParams.set("sid", existingId);
    }
    return url.toString();
  }

  let iframeLoaded = false;

  bubble.addEventListener("click", () => {
    if (!iframeLoaded) {
      iframe.src = buildIframeSrc();
      iframeLoaded = true;
    }
    iframe.style.display = iframe.style.display === "none" ? "block" : "none";
  });

  // --- Listen for a newly created session ID from the iframe ---
  window.addEventListener("message", (event) => {
    if (event.origin !== apiBase) return;
    if (event.data && event.data.type === "liquidlab-session-created") {
      localStorage.setItem(STORAGE_KEY, event.data.sessionId);
    }
  });

  document.body.appendChild(iframe);
  document.body.appendChild(bubble);
})();