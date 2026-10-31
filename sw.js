// the app moved to a new address: empty the old caches and step aside, so the next start shows the move page
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(k => Promise.all(k.map(n => caches.delete(n))))
  .then(() => self.registration.unregister()).then(() => self.clients.matchAll({ type: 'window' }))
  .then(cs => cs.forEach(c => c.navigate(c.url).catch(() => {})))));
