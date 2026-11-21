const DEFAULT_API = "https://secframework.vercel.app";

async function getApiBase() {
  const saved = await chrome.storage.sync.get({ apiBase: DEFAULT_API });
  return String(saved.apiBase || DEFAULT_API).replace(/\/$/, "");
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "HG_SCREEN_PROMPT") {
    (async () => {
      try {
        const base = await getApiBase();
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 65000);
        try {
          const response = await fetch(base + "/api/screen", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: String(message.text || "").slice(0, 4000), role: "unknown", source: "user", verified: false }),
            signal: controller.signal
          });
          if (!response.ok) throw new Error("API returned HTTP " + response.status);
          sendResponse({ ok: true, result: await response.json() });
        } finally {
          clearTimeout(timeout);
        }
      } catch (error) {
        sendResponse({ ok: false, error: error.name === "AbortError" ? "The API took too long to respond. A free service may be waking up." : error.message });
      }
    })();
    return true;
  }
  if (message?.type === "HG_TEST_API") {
    (async () => {
      try {
        const base = await getApiBase();
        const response = await fetch(base + "/api/health", { method: "GET" });
        if (!response.ok) throw new Error("API returned HTTP " + response.status);
        sendResponse({ ok: true, health: await response.json() });
      } catch (error) {
        sendResponse({ ok: false, error: error.message });
      }
    })();
    return true;
  }
  if (message?.type === "HG_OPEN_DASHBOARD") {
    (async () => {
      const base = await getApiBase();
      await chrome.tabs.create({ url: base + "/" });
      sendResponse({ ok: true });
    })();
    return true;
  }
});
