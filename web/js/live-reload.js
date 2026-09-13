/**
 * LIVE sync: when Cursor saves files, reload this window automatically.
 */
(function liveReload() {
  const badge = document.createElement("div");
  badge.id = "liveSyncBadge";
  badge.textContent = "LIVE sync on";
  badge.style.cssText = [
    "position:fixed",
    "right:12px",
    "bottom:12px",
    "z-index:99999",
    "padding:6px 10px",
    "border-radius:999px",
    "font:600 11px Manrope,system-ui,sans-serif",
    "letter-spacing:0.04em",
    "color:#06201e",
    "background:linear-gradient(135deg,#5ed1cb,#1aa6a0)",
    "box-shadow:0 8px 24px rgba(0,0,0,.35)",
    "opacity:0.92",
    "pointer-events:none",
  ].join(";");
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(badge));

  let es;
  const connect = () => {
    try {
      es = new EventSource("/api/dev/reload-stream");
    } catch (err) {
      console.warn("live-reload SSE unavailable", err);
      pollFallback();
      return;
    }
    es.addEventListener("reload", () => {
      badge.textContent = "Syncing…";
      // brief delay so multi-file saves settle
      setTimeout(() => location.reload(), 250);
    });
    es.onerror = () => {
      badge.textContent = "LIVE reconnecting";
      es.close();
      setTimeout(connect, 1200);
    };
  };

  let last = "";
  const pollFallback = () => {
    setInterval(async () => {
      try {
        const res = await fetch("/api/dev/version?_=" + Date.now(), { cache: "no-store" });
        const data = await res.json();
        if (!last) last = data.fingerprint;
        else if (data.fingerprint !== last) location.reload();
      } catch (_) {
        /* server restarting */
      }
    }, 1000);
  };

  connect();
})();
