(function () {
  const scriptTag = document.currentScript;
  const apiBase = scriptTag.getAttribute("data-api");
  const tenantId = scriptTag.getAttribute("data-tenant-id");
  const STORAGE_KEY = "liquidlab_chat_session_id";

  const API_ORIGIN = new URL(apiBase).origin;

  const bubbleWrap = document.createElement("div");
  bubbleWrap.style.cssText = `
    position: fixed; bottom: 20px; right: 20px;
    width: 56px; height: 56px; z-index: 999999;
  `;

  const halo = document.createElement("div");
  halo.style.cssText = `
    position: absolute; inset: -8px; border-radius: 50%;
    border: 1px solid rgba(201, 162, 75, 0.35);
    animation: liquidlab-halo 2.6s ease-out infinite;
    pointer-events: none;
  `;

  const haloStyle = document.createElement("style");
  haloStyle.textContent = `
    @keyframes liquidlab-halo {
      0% { transform: scale(0.92); opacity: 0.7; }
      100% { transform: scale(1.35); opacity: 0; }
    }
  `;
  document.head.appendChild(haloStyle);

  const bubble = document.createElement("button");
  bubble.innerHTML = `
    <svg class="liquidlab-ico-open" viewBox="0 0 24 24" fill="none" stroke="#1A1305" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="26" height="26">
      <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
    </svg>
    <svg class="liquidlab-ico-close" viewBox="0 0 24 24" fill="none" stroke="#1A1305" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="22" height="22">
      <line x1="18" y1="6" x2="6" y2="18"></line>
      <line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>
  `;
  bubble.style.cssText = `
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%;
    background: linear-gradient(155deg, #E8C36B, #C9A24B 70%);
    color: #1A1305; border: none;
    cursor: pointer;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(201, 162, 75, 0.25);
    transition: transform .15s ease, filter .15s ease, box-shadow .15s ease;
  `;
  const iconStyle = document.createElement("style");
  iconStyle.textContent = `
    .liquidlab-ico-open, .liquidlab-ico-close {
      position: absolute; transition: opacity .15s ease, transform .15s ease;
    }
    .liquidlab-ico-close { opacity: 0; transform: scale(0.6) rotate(-45deg); }
    .liquidlab-bubble-open .liquidlab-ico-open { opacity: 0; transform: scale(0.6) rotate(45deg); }
    .liquidlab-bubble-open .liquidlab-ico-close { opacity: 1; transform: scale(1) rotate(0deg); }
  `;
  document.head.appendChild(iconStyle);
  bubble.addEventListener("mouseenter", () => {
    bubble.style.transform = "scale(1.06)";
    bubble.style.filter = "brightness(1.06)";
  });
  bubble.addEventListener("mouseleave", () => {
    bubble.style.transform = "scale(1)";
    bubble.style.filter = "brightness(1)";
  });

  bubble.setAttribute("aria-label", "Open chat");

  bubbleWrap.appendChild(halo);
  bubbleWrap.appendChild(bubble);

  const iframe = document.createElement("iframe");
  iframe.style.cssText = `
    position: fixed; bottom: 90px; right: 20px;
    width: 360px; height: 520px; border: none;
    border-radius: 10px; box-shadow: 0 24px 70px rgba(0,0,0,0.5);
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
    halo.style.opacity = opening ? "0" : "1";
    bubble.classList.toggle("liquidlab-bubble-open", opening);

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
      halo.style.opacity = "1";
      bubble.classList.remove("liquidlab-bubble-open");
    }
  });

  document.body.appendChild(iframe);
  document.body.appendChild(bubbleWrap);
})();