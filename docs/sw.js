/* Squad Room service worker — offline app shell. Bump CACHE to force an update. */
const CACHE = "squadroom-v2";
const ASSETS = ["live_draft.html","index.html","dashboard.html","manifest.webmanifest",
  "icon-192.png","icon-512.png","icon-512-maskable.png","apple-touch-icon.png"];
self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;
  const isDoc = req.mode === "navigate" || req.destination === "document";
  if (isDoc) {                                   // network-first: fresh when online, cached offline
    e.respondWith(fetch(req).then(res => { const cp = res.clone(); caches.open(CACHE).then(c => c.put(req, cp)); return res; })
      .catch(() => caches.match(req).then(h => h || caches.match("live_draft.html"))));
  } else {                                        // cache-first for assets, fonts, images
    e.respondWith(caches.match(req).then(hit => hit || fetch(req).then(res => {
      try { const cp = res.clone(); caches.open(CACHE).then(c => c.put(req, cp)); } catch (_) {} return res;
    }).catch(() => hit)));
  }
});
