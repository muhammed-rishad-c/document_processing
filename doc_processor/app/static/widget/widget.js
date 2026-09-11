(function () {
  const scriptTag = document.currentScript;
  const apiBase = scriptTag.getAttribute("data-api");
  const tenantId = scriptTag.getAttribute("data-tenant-id");
  const STORAGE_KEY = "liquidlab_chat_session_id";

  const API_ORIGIN = new URL(apiBase).origin;

  const bubble = document.createElement("button");
  bubble.textContent = "💬";
  bubble.style.cssText = `
    position: fixed; bottom: 20px; right: 20px;
    width: 56px; height: 56px; border-radius: 50%;
    background: #333; color: white; border: none;
    font-size: 24px; cursor: pointer; z-index: 999999;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  `;

  const iframe = document.createElement("iframe");
  iframe.style.cssText = `
    position: fixed; bottom: 90px; right: 20px;
    width: 360px; height: 520px; border: none;
    border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    z-index: 999999; display: none;
  `;

  function buildIframeSrc() {
    const existingId = localStorage.getItem(STORAGE_KEY);
    const url = new URL(`${API_ORIGIN}/widget-ui/chat.html`);
    if (existingId) {
      url.searchParams.set("sid", existingId);
    }
    if (tenantId) {
      url.searchParams.set("tenant", tenantId);
    }
    url.searchParams.set("embed_origin", window.location.origin);
    return url.toString();
  }

  let iframeLoaded = false;
  let iframeReady = false;

  iframe.addEventListener("load", () => {
    iframeReady = true;
  });

  function sendResumeMessage() {
    iframe.contentWindow.postMessage({ type: "liquidlab-resume-chat" }, API_ORIGIN);
  }

  bubble.addEventListener("click", () => {
    if (!iframeLoaded) {
      iframe.src = buildIframeSrc();
      iframeLoaded = true;
    }
    const opening = iframe.style.display === "none";
    iframe.style.display = opening ? "block" : "none";

    if (opening) {
      if (iframeReady) {
        // Iframe already finished loading from a previous open — safe to
        // send immediately.
        sendResumeMessage();
      } else {
        // Iframe is still navigating (first open, or page just loaded) —
        // wait for it to actually finish before sending, otherwise the
        // browser rejects the message as a same-origin/about:blank mismatch.
        iframe.addEventListener("load", sendResumeMessage, { once: true });
      }
    }
  });

  window.addEventListener("message", (event) => {
    if (event.origin !== API_ORIGIN) return;
    if (event.data && event.data.type === "liquidlab-session-created") {
      localStorage.setItem(STORAGE_KEY, event.data.sessionId);
    }
    if (event.data && event.data.type === "liquidlab-close-widget") {
      iframe.style.display = "none";
    }
  });

  document.body.appendChild(iframe);
  document.body.appendChild(bubble);
})();