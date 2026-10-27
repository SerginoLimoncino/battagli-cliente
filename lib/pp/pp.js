// Text reader for photos (PP-OCRv4 models through onnxruntime-web): finds the lines of text, reads each one,
// and matches handwritten or printed catalogue codes (12.3456.78) even when some digits are badly written.
// window.PPOCR = { ready(), read(canvas) -> { lines:[{text, digits, box}], probs } , matchCodes(lines, codes) }
(function () {
  let readyP = null, det = null, rec = null, KEYS = null, CLS = null;
  function loadScript(src) { return new Promise((ok, ko) => { const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = ko; document.head.appendChild(s); }); }
  function ready() {
    if (readyP) return readyP;
    const B = new URL(window.PP_BASE || 'lib/pp/', location.href).href;
    readyP = (async () => {
      if (!window.ort) await loadScript(B + 'ort.wasm.min.js');
      // the loader module is shipped as .js: some app servers only know .js as JavaScript
      ort.env.wasm.numThreads = 1; ort.env.wasm.wasmPaths = { mjs: B + 'ort-wasm-simd-threaded.js', wasm: B + 'ort-wasm-simd-threaded.wasm' };
      const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
      const [d, r, k] = await Promise.all([ort.InferenceSession.create(B + 'det.onnx', opt), ort.InferenceSession.create(B + 'rec.onnx', opt), fetch(B + 'keys.txt').then(x => { if (!x.ok) throw new Error('keys'); return x.text(); })]);
      det = d; rec = r;
      KEYS = ['', ...k.split('\n').filter((l, i, a) => i < a.length - 1 || l !== ''), ' '];
      // class groups used by the code matcher
      CLS = { digit: [], sep: [] };
      KEYS.forEach((c, i) => { if (/^[0-9]$/.test(c)) CLS.digit[+c] = i; else if (/^[.,\-·:_/ ]$/.test(c)) CLS.sep.push(i); });
      return true;
    })();
    readyP.catch(() => { readyP = null; });
    return readyP;
  }

  // ---------- line finder ----------
  async function findLines(src, maxSide) {
    const s = Math.min(1, maxSide / Math.max(src.width, src.height));
    const W = Math.max(32, Math.round(src.width * s / 32) * 32), H = Math.max(32, Math.round(src.height * s / 32) * 32);
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const x = c.getContext('2d', { willReadFrequently: true }); x.imageSmoothingQuality = 'high'; x.drawImage(src, 0, 0, W, H);
    const px = x.getImageData(0, 0, W, H).data, n = W * H, inp = new Float32Array(3 * n);
    const mean = [0.406, 0.456, 0.485], std = [0.225, 0.224, 0.229]; // BGR order, as the model was trained
    for (let i = 0; i < n; i++) { const r = px[i * 4] / 255, g = px[i * 4 + 1] / 255, b = px[i * 4 + 2] / 255; inp[i] = (b - mean[0]) / std[0]; inp[n + i] = (g - mean[1]) / std[1]; inp[2 * n + i] = (r - mean[2]) / std[2]; }
    const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', inp, [1, 3, H, W]) });
    const prob = out[det.outputNames[0]].data;
    const m = new Uint8Array(n); for (let i = 0; i < n; i++) m[i] = prob[i] > 0.3 ? 1 : 0;
    const lab = new Int32Array(n), boxes = []; let id = 0;
    for (let i = 0; i < n; i++) {
      if (!m[i] || lab[i]) continue; id++;
      const st = [i], pts = []; lab[i] = id; let sc = 0;
      while (st.length) { const p = st.pop(); pts.push(p); sc += prob[p]; const xx = p % W;
        if (xx > 0 && m[p - 1] && !lab[p - 1]) { lab[p - 1] = id; st.push(p - 1); }
        if (xx < W - 1 && m[p + 1] && !lab[p + 1]) { lab[p + 1] = id; st.push(p + 1); }
        if (p >= W && m[p - W] && !lab[p - W]) { lab[p - W] = id; st.push(p - W); }
        if (p < n - W && m[p + W] && !lab[p + W]) { lab[p + W] = id; st.push(p + W); } }
      if (pts.length < 12 || sc / pts.length < 0.5) continue;
      // smallest rotated rectangle (angles -45..45)
      let mx = 0, my = 0; for (const p of pts) { mx += p % W; my += (p / W) | 0; } mx /= pts.length; my /= pts.length;
      let best = 1e18, B = null;
      for (let a = -45; a <= 45; a += 3) {
        const t = a * Math.PI / 180, ca = Math.cos(t), sa = Math.sin(t); let u0 = 1e9, u1 = -1e9, v0 = 1e9, v1 = -1e9;
        const step = pts.length > 3000 ? 3 : 1;
        for (let j = 0; j < pts.length; j += step) { const p = pts[j], dx = p % W - mx, dy = ((p / W) | 0) - my, u = dx * ca + dy * sa, v = -dx * sa + dy * ca; if (u < u0) u0 = u; if (u > u1) u1 = u; if (v < v0) v0 = v; if (v > v1) v1 = v; }
        const ar = (u1 - u0 + 1) * (v1 - v0 + 1); if (ar < best) { best = ar; B = { t, u0, u1, v0, v1 }; }
      }
      let w = B.u1 - B.u0 + 1, h = B.v1 - B.v0 + 1, ang = B.t;
      if (h > w * 1.5) { [w, h] = [h, w]; ang += Math.PI / 2; } // vertical text line
      const cu = (B.u0 + B.u1) / 2, cv = (B.v0 + B.v1) / 2, ca = Math.cos(B.t), sa = Math.sin(B.t);
      // unclip: grow by area * 1.5 / perimeter on every side
      const d = (w * h * 1.5) / (2 * (w + h));
      // back to source pixels (the map was stretched to multiples of 32)
      const sx = src.width / W, sy = src.height / H, k = (sx + sy) / 2;
      boxes.push({ cx: (mx + cu * ca - cv * sa) * sx, cy: (my + cu * sa + cv * ca) * sy, w: (w + 2 * d) * k, h: (h + 2 * d) * k, ang });
    }
    return boxes.sort((a, b) => a.cy - b.cy || a.cx - b.cx);
  }
  // Handwriting often comes out in pieces ("01" ... "0577.04"): pieces side by side on the same line,
  // of about the same height, are also read joined together.
  function joinPieces(boxes) {
    const flat = boxes.filter(b => Math.abs(Math.sin(b.ang)) < 0.35).map(b => ({ b, x0: b.cx - b.w / 2, x1: b.cx + b.w / 2 })).sort((p, q) => p.x0 - q.x0);
    const out = [], used = new Set();
    for (let i = 0; i < flat.length; i++) {
      let cur = [flat[i]];
      for (let j = i + 1; j < flat.length; j++) {
        const L = cur[cur.length - 1], R = flat[j], h = Math.max(L.b.h, R.b.h), hr = Math.max(L.b.h, R.b.h) / Math.min(L.b.h, R.b.h);
        const gap = R.x0 - L.x1;
        if (Math.abs(L.b.cy - R.b.cy) < 0.45 * h && hr < 1.9 && gap > -0.5 * h && gap < 1.3 * h) cur.push(R);
      }
      if (cur.length < 2) continue;
      const key = cur.map(c => flat.indexOf(c)).join(','); if (used.has(key)) continue; used.add(key);
      const x0 = Math.min(...cur.map(c => c.x0)), x1 = Math.max(...cur.map(c => c.x1)), W = cur.reduce((a, c) => a + c.b.w, 0);
      out.push({ cx: (x0 + x1) / 2, cy: cur.reduce((a, c) => a + c.b.cy * c.b.w, 0) / W, w: x1 - x0, h: Math.max(...cur.map(c => c.b.h)), ang: cur.reduce((a, c) => a + c.b.ang * c.b.w, 0) / W, joined: cur.length });
    }
    // keep only the longest chains (a chain that is part of a longer one adds nothing)
    return out.filter(o => !out.some(p => p !== o && p.joined > o.joined && p.cx - p.w / 2 <= o.cx - o.w / 2 + 1 && p.cx + p.w / 2 >= o.cx + o.w / 2 - 1 && Math.abs(p.cy - o.cy) < o.h / 2));
  }

  // ---------- line reader ----------
  function cropLine(src, b, flip) {
    const H = 48, Wd = Math.max(16, Math.min(1600, Math.round(H * b.w / b.h)));
    const c = document.createElement('canvas'); c.width = Wd; c.height = H;
    const x = c.getContext('2d', { willReadFrequently: true }); x.imageSmoothingQuality = 'high';
    x.translate(Wd / 2, H / 2); x.rotate(-b.ang + (flip ? Math.PI : 0)); x.scale(H / b.h, H / b.h); x.drawImage(src, -b.cx, -b.cy);
    return c;
  }
  async function readLine(c) {
    const W = c.width, H = c.height, px = c.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, W, H).data, n = W * H, inp = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { inp[i] = px[i * 4 + 2] / 127.5 - 1; inp[n + i] = px[i * 4 + 1] / 127.5 - 1; inp[2 * n + i] = px[i * 4] / 127.5 - 1; }
    const out = await rec.run({ [rec.inputNames[0]]: new ort.Tensor('float32', inp, [1, 3, H, W]) });
    const o = out[rec.outputNames[0]], [, T, C] = o.dims, P = o.data;
    let text = '', last = 0, conf = 0, nc = 0;
    for (let t = 0; t < T; t++) { let bi = 0, bv = -1; for (let k = 0; k < C; k++) if (P[t * C + k] > bv) { bv = P[t * C + k]; bi = k; } if (bi && bi !== last) { text += KEYS[bi] || ''; conf += bv; nc++; } last = bi; }
    return { text, conf: nc ? conf / nc : 0, T, C, P };
  }

  // ---------- code matcher ----------
  // For one line: frames as log-probabilities of blank (blank + separators), digits 0-9, and "anything".
  function frames(r) {
    const { T, C, P } = r, out = [];
    for (let t = 0; t < T; t++) {
      const o = t * C; let bl = P[o]; for (const k of CLS.sep) bl += P[o + k];
      let any = 0; for (let k = 0; k < C; k++) if (P[o + k] > any) any = P[o + k];
      const dg = new Float32Array(10); for (let d = 0; d < 10; d++) dg[d] = Math.log(P[o + CLS.digit[d]] + 1e-6);
      out.push({ bl: Math.log(bl + 1e-6), any: Math.log(any + 1e-6), dg });
    }
    return out;
  }
  // Best path that reads the 8 digits somewhere in the line (anything before and after);
  // cost = how much less likely than what the reader saw on its own. 0 = read exactly.
  function codeCost(F, code) {
    const L = code.length, S = 2 * L + 1, NEG = -1e9;
    let pre = 0, cur = new Float64Array(S).fill(NEG), post = NEG, free = 0;
    for (let t = 0; t < F.length; t++) {
      const f = F[t], nx = new Float64Array(S).fill(NEG);
      for (let s = 0; s < S; s++) {
        const isBl = s % 2 === 0, e = isBl ? f.bl : f.dg[code.charCodeAt((s - 1) >> 1) - 48];
        let b = cur[s];
        if (s > 0 && cur[s - 1] > b) b = cur[s - 1];
        if (!isBl && s > 1 && code[(s - 1) >> 1] !== code[(s - 3) >> 1] && cur[s - 2] > b) b = cur[s - 2];
        if (s <= 1 && pre > b) b = pre;
        nx[s] = b + e;
      }
      post = Math.max(post, cur[S - 1], cur[S - 2]) + f.any;
      pre += f.any; free += f.any; cur = nx;
    }
    const best = Math.max(post, cur[S - 1], cur[S - 2]);
    return free - best;
  }
  // For each line holding a number: the catalogue codes that fit it best, [{ line, n, c: [{ code, cost }] }].
  // cost 0 = read exactly; every point more = less likely (about one badly written digit = 3-6).
  function matchCodes(lines, codes, keep = 4) {
    const res = [];
    for (const ln of lines) {
      const n = (ln.digits || '').replace(/\D/g, '').length;
      if (!ln.F || n < 5) continue;
      const top = [];
      for (const c of codes) {
        const k = codeCost(ln.F, c);
        if (top.length < keep || k < top[top.length - 1].cost) { top.push({ code: c, cost: k }); top.sort((a, b) => a.cost - b.cost); if (top.length > keep) top.pop(); }
      }
      res.push({ line: ln.text, conf: ln.conf, n, c: top });
    }
    return res;
  }
  // digits-only reading of a line (letters that look like digits are not allowed)
  function digitsOnly(F) {
    let s = '', last = -1;
    for (const f of F) { let bi = -1, bv = f.bl; for (let d = 0; d < 10; d++) if (f.dg[d] > bv) { bv = f.dg[d]; bi = d; } if (bi >= 0 && bi !== last) s += bi; else if (bi < 0 && last >= 0) s += ' '; last = bi; }
    return s.replace(/\s+/g, ' ').trim();
  }

  async function read(src, opt = {}) {
    await ready();
    // the line finder may already have looked at this picture (the quick look for documents)
    const found = opt.boxes || await findLines(src, opt.maxSide || window.PP_MAXSIDE || 960), lines = [];
    const boxes = found.concat(joinPieces(found));
    // lines already read in an earlier pass are not read again
    const same = (a, b) => Math.abs(a.cx - b.cx) < 0.25 * a.w && Math.abs(a.cy - b.cy) < 0.4 * a.h && a.w / b.w > 0.8 && a.w / b.w < 1.25;
    for (const b of boxes.slice(0, opt.maxLines || 60)) {
      if (opt.isLive && !opt.isLive()) break;
      if (b.h < 6 || b.w < 8) continue;
      if (opt.skip && opt.skip.some(x => same(x.box, b))) continue;
      let r = await readLine(cropLine(src, b, false));
      // a line standing up can be read from either side: keep the side the reader is surest about
      if (Math.abs(Math.sin(b.ang)) > 0.7 || r.conf < 0.6) { const r2 = await readLine(cropLine(src, b, true)); if (r2.conf > r.conf) r = r2; }
      const F = frames(r); lines.push({ text: r.text, conf: r.conf, digits: digitsOnly(F), F, box: b });
    }
    return lines;
  }
  window.PPOCR = { ready, read, matchCodes, codeCost, findLines, cropLine, readLine, frames };
})();
