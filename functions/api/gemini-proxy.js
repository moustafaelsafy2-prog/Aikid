// functions/api/gemini-proxy.js
// Cloudflare Pages Function — Gemini proxy (Cloudflare runtime port of
// netlify/functions/gemini-proxy.js). Same behavior, same request/response
// contract as the frontend expects; adapted from Netlify's
// `exports.handler(event)` shape to Cloudflare's `onRequestPost({request, env})`.
//
// - Pro-first model pool with smart fallbacks
// - Strict guardrails (UAE child-safety aware wording layer)
// - SSE streaming with heartbeats
// - Robust JSON enforcement for plan/qa/expect=json
// - Media sanitization (type/size)
// - Retries with exponential backoff + jitter
// - In-memory rate limiting (best-effort, warm isolates only)
// - Timeouts + graceful upstream error shaping

/* ================== Config ================== */
const MAX_TRIES = 3;
const BASE_BACKOFF_MS = 650;
const DEFAULT_TIMEOUT_MS = 28000;
const MAX_INLINE_BYTES = 15 * 1024 * 1024; // 15MB data URLs
const MAX_MESSAGES = 64;
const MAX_PARTS_PER_MSG = 16;
const MAX_TEXT_CHARS = 24_000; // soft clamp for prompt bloat
const RATE_WINDOW_MS = 10 * 60 * 1000; // 10 minutes
const RATE_MAX_REQ = 60;               // 60 req / 10 min per client
const TRUSTED_ORIGINS = [/^https?:\/\/localhost(?::\d+)?$/i];

/* ================== Model Strategy ================== */
const MODEL_POOL = [
  "gemini-1.5-pro",
  "gemini-1.5-pro-latest",
  "gemini-2.0-flash",
  "gemini-1.5-flash",
  "gemini-2.0-flash-exp"
];

/* ================== Media allowlists ================== */
const ALLOWED_IMAGE = /^image\/(png|jpe?g|webp|gif|bmp|svg\+xml)$/i;
const ALLOWED_AUDIO = /^audio\/(webm|ogg|mp3|mpeg|wav|m4a|aac|3gpp|3gpp2|mp4)$/i;

/* ================== In-memory rate limiting ================== */
// Best-effort only: each Cloudflare isolate has its own memory, so this
// does not provide a global rate limit across the edge network. Same
// caveat applies to the Netlify version.
const bucket = new Map(); // key -> { count, resetAt }
function clientKey(headers) {
  const ip = headers.get("cf-connecting-ip") || headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "0.0.0.0";
  const ua = headers.get("user-agent") || "na";
  return `${ip}::${ua.slice(0, 80)}`;
}
function rateCheck(headers) {
  const k = clientKey(headers);
  const now = Date.now();
  const e = bucket.get(k);
  if (!e || now > e.resetAt) {
    bucket.set(k, { count: 1, resetAt: now + RATE_WINDOW_MS });
    return { ok: true };
  }
  if (e.count >= RATE_MAX_REQ) return { ok: false, retryAfter: Math.ceil((e.resetAt - now) / 1000) };
  e.count++;
  return { ok: true };
}

/* ================== Entry ================== */
export async function onRequestOptions({ request }) {
  return new Response(null, { status: 204, headers: baseHeaders(request) });
}

export async function onRequestPost({ request, env }) {
  const reqId = (Math.random().toString(36).slice(2) + Date.now().toString(36)).toUpperCase();
  const started = Date.now();
  const bHeaders = baseHeaders(request, reqId);

  const rl = rateCheck(request.headers);
  if (!rl.ok) {
    return respond(429, bHeaders, {
      error: "Rate limit exceeded",
      requestId: reqId,
      retry_after_seconds: rl.retryAfter
    });
  }

  const API_KEY = env.GEMINI_API_KEY;
  if (!API_KEY) return respond(500, bHeaders, { error: "Missing GEMINI_API_KEY", requestId: reqId });

  // ---------- Parse & validate input
  let body = {};
  try { body = await request.json(); }
  catch { return respond(400, bHeaders, { error: "Invalid JSON body", requestId: reqId }); }

  let {
    system = "",
    messages = [],
    model = "auto",
    mode = "default",
    force_lang,
    guard_level = "strict",
    stream = false,
    long = true,
    max_chunks = 4,
    timeout_ms = DEFAULT_TIMEOUT_MS,
    include_raw = false,
    expect = ""
  } = body || {};

  timeout_ms = clamp(timeout_ms, 1000, 29000, DEFAULT_TIMEOUT_MS);

  // ---------- Sanitize messages (count/lengths)
  if (!Array.isArray(messages)) messages = [];
  if (messages.length > MAX_MESSAGES) messages = messages.slice(-MAX_MESSAGES);
  messages = messages.map(m => ({
    ...m,
    content: (typeof m?.content === "string" ? m.content : "").slice(0, MAX_TEXT_CHARS),
    images: Array.isArray(m?.images) ? m.images.slice(0, MAX_PARTS_PER_MSG) : undefined,
    audio: m?.audio || undefined
  }));

  // ---------- Language & guardrails
  const preview = sample(messages);
  const lang = chooseLang(force_lang, preview);
  const rails = buildGuardrails({ lang, level: guard_level, imageMode: (mode === "image_brief") });

  // ---------- Normalize messages (+ media)
  let contents = normalizeMessages(messages, rails);

  if (!contents.length) {
    const seed = (typeof system === "string" && system.trim()) ? system.trim() : "ابدأ الآن.";
    contents = [{ role: "user", parts: [{ text: wrapPrompt(seed, rails) }] }];
  }

  const generationConfig = tuneGeneration(mode, expect);
  const safetySettings = buildSafety(guard_level);

  const systemInstruction = (system && typeof system === "string" && system.trim())
    ? { role: "system", parts: [{ text: system }] }
    : undefined;

  const candidates = (model === "auto" || !model)
    ? [...MODEL_POOL]
    : Array.from(new Set([model, ...MODEL_POOL]));

  // ---------- Streaming path (SSE)
  if (stream) {
    const sseHeaders = {
      ...bHeaders,
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "Connection": "keep-alive"
    };
    for (let mi = 0; mi < candidates.length; mi++) {
      const m = candidates[mi];
      const url = makeUrl(m, true, API_KEY);
      const reqBody = JSON.stringify({ contents, generationConfig, safetySettings, ...(systemInstruction ? { systemInstruction } : {}) });

      const once = await tryStreamOnce(url, reqBody, timeLeft(started, timeout_ms), bHeaders);
      if (once.ok) {
        const reader = once.response.body.getReader();
        const encoder = new TextEncoder();
        const meta = encoder.encode(`event: meta\ndata: ${JSON.stringify({ requestId: reqId, model: m, lang })}\n\n`);

        const streamBody = new ReadableStream({
          async start(controller) {
            controller.enqueue(meta);
            let hb = Date.now();
            let buffer = "";
            try {
              while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += new TextDecoder().decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";
                for (const line of lines) {
                  const trimmed = line.trim();
                  if (!trimmed) continue;
                  controller.enqueue(encoder.encode(`event: chunk\ndata: ${trimmed}\n\n`));
                }
                if (Date.now() - hb > 5000) {
                  hb = Date.now();
                  controller.enqueue(encoder.encode(`event: ping\ndata: {}\n\n`));
                }
              }
              controller.enqueue(encoder.encode(`event: end\ndata: ${JSON.stringify({ model: m, took_ms: Date.now() - started })}\n\n`));
            } catch (e) {
              controller.error(e);
              return;
            }
            controller.close();
          }
        });

        return new Response(streamBody, { status: 200, headers: sseHeaders });
      }
      if (mi === candidates.length - 1) {
        return once.errorResp || respond(502, bHeaders, { error: "All models failed (stream)", requestId: reqId, lang });
      }
    }
  }

  // ---------- Non-stream with fallback + auto-continue
  for (let mi = 0; mi < candidates.length; mi++) {
    const m = candidates[mi];
    const url = makeUrl(m, false, API_KEY);
    const makeBody = () => JSON.stringify({
      contents, generationConfig, safetySettings, ...(systemInstruction ? { systemInstruction } : {})
    });

    const first = await tryJSONOnce(url, makeBody(), timeLeft(started, timeout_ms), include_raw);
    if (!first.ok) {
      if ((mode === "plan" || mode === "qa" || expect === "json") && first.error && /Empty\/blocked|safety/i.test(first.error.error || "")) {
        const strictCfg = tuneGeneration("qa", "json");
        const altBody = () => JSON.stringify({ contents, generationConfig: strictCfg, safetySettings, ...(systemInstruction ? { systemInstruction } : {}) });
        const second = await tryJSONOnce(url, altBody(), timeLeft(started, timeout_ms), include_raw);
        if (second.ok) return ok(bHeaders, reqId, finalize(second.text, lang, expect, mode), m, lang, started, second.usage);
      }
      if (mi === candidates.length - 1) {
        return respond(first.statusCode || 502, bHeaders, {
          error: "Upstream error",
          requestId: reqId,
          modelTried: m,
          upstream: first.error || { error: "Unknown upstream error" }
        });
      }
      continue;
    }

    let text = first.text;
    let chunks = 1;

    while (long && chunks < clamp(max_chunks, 1, 12, 4) && shouldContinue(text) && timeLeft(started, timeout_ms) > 2500) {
      contents.push({ role: "model", parts: [{ text }] });
      contents.push({ role: "user", parts: [{ text: continuePrompt(lang) }] });
      const next = await tryJSONOnce(url, makeBody(), timeLeft(started, timeout_ms), false);
      if (!next.ok) break;
      const append = dedupeContinuation(text, next.text);
      text += (append ? ("\n" + append) : "");
      chunks++;
    }

    return ok(bHeaders, reqId, finalize(text, lang, expect, mode), m, lang, started, first.usage);
  }

  return respond(500, bHeaders, { error: "Unknown failure", requestId: reqId, lang });
}

/* ================== Helpers ================== */
function baseHeaders(request, reqId) {
  const h = {
    "Access-Control-Allow-Origin": allowOrigin(request.headers),
    "Access-Control-Allow-Headers": "Content-Type, X-Request-ID",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Vary": "Origin"
  };
  if (reqId) h["X-Request-ID"] = reqId;
  return h;
}
function respond(code, headers, obj) {
  return new Response(JSON.stringify(obj || {}), { status: code, headers: { ...headers, "Content-Type": "application/json; charset=utf-8" } });
}
function ok(h, id, text, model, lang, started, usage) {
  return new Response(JSON.stringify({ text, model, lang, requestId: id, took_ms: Date.now() - started, usage }),
    { status: 200, headers: { ...h, "Content-Type": "application/json; charset=utf-8" } });
}
function makeUrl(model, isStream, key) {
  const base = "https://generativelanguage.googleapis.com/v1beta/models";
  const method = isStream ? "streamGenerateContent" : "generateContent";
  return `${base}/${encodeURIComponent(model)}:${method}?key=${key}`;
}
function timeLeft(start, total) { return Math.max(0, total - (Date.now() - start)); }
function clamp(n, min, max, def) { const v = Number.isFinite(+n) ? +n : def; return Math.max(min, Math.min(max, v)); }
function hasArabic(s) { return /[؀-ۿ]/.test(s || ""); }
function chooseLang(force, sample) { if (force === "ar" || force === "en") return force; return hasArabic(sample) ? "ar" : "en"; }
function mirrorLanguage(text, lang) {
  if (!text) return text;
  if (lang === "ar" && hasArabic(text)) return text;
  if (lang === "en" && !hasArabic(text)) return text;
  return (lang === "ar") ? `**ملاحظة:** أجب بالعربية فقط.\n\n${text}` : `**Note:** Respond in English only.\n\n${text}`;
}
function sample(msgs) { return (Array.isArray(msgs) ? msgs.map(m => (m?.content || "")).join("\n") : "").slice(0, 4000); }
function allowOrigin(headers) {
  const origin = headers.get("origin") || "*";
  if (origin === "*") return "*";
  if (TRUSTED_ORIGINS.some(rx => rx.test(origin))) return origin;
  return "*"; // fallback permissive; tighten via platform headers for prod
}

/* ---------- Guardrails & wrapping ---------- */
function buildGuardrails({ lang, level = "strict", imageMode = false }) {
  const L = (lang === "ar")
    ? {
        mirror: "أجب حصراً بالعربية؛ لا تخلط لغتين ولا تضف ترجمات.",
        brief: "اختصر الحشو وقدّم خطوات عملية واضحة.",
        img: "إن وُجدت صور: 3–5 نقاط تنفيذية + خطوة فورية واحدة. بلا مقدمات.",
        strict: "لا تختلق. عند الشك اطلب توضيحاً. التزم بمعايير سلامة الطفل وقوانين دولة الإمارات."
      }
    : {
        mirror: "Answer strictly in English; don't mix languages or add translations.",
        brief: "Be concise and actionable.",
        img: "If images present: 3–5 precise bullets + one immediate step. No preamble.",
        strict: "No fabrication. Ask for missing info. Adhere to child-safety norms applicable in the UAE."
      };
  const out = [L.mirror, L.brief];
  if (imageMode) out.push(L.img);
  if (level !== "relaxed") out.push(L.strict);
  return out.join("\n");
}
function wrapPrompt(text, guard) {
  const head = "تعليمات حراسة (اتّبع بدقة):";
  return `${head}\n${guard}\n\n---\n${text || ""}`;
}

/* ---------- Messages & media normalization ---------- */
function normalizeMessages(messages, guard) {
  const safeRole = (r) => (r === "user" || r === "model" || r === "system") ? r : "user";
  let injected = false;
  return (messages || [])
    .filter(m => m && (typeof m.content === "string" || m.images || m.audio))
    .map(m => {
      const parts = [];
      const raw = (typeof m.content === "string" && m.content.trim()) ? m.content : "";
      if (!injected && m.role === "user") {
        parts.push({ text: wrapPrompt(raw, buildGuardrails({ lang: chooseLang(undefined, raw), level: "strict" })) });
        injected = true;
      } else if (raw) {
        parts.push({ text: raw });
      }
      if (Array.isArray(m.images)) {
        for (const item of m.images) {
          const { mime, data } = coerceData(item);
          if (mime && data && ALLOWED_IMAGE.test(mime) && approxBase64Bytes(data) <= MAX_INLINE_BYTES) {
            parts.push({ inline_data: { mime_type: mime, data } });
          }
        }
      }
      if (m.audio) {
        const { mime, data } = coerceData(m.audio);
        if (mime && data && ALLOWED_AUDIO.test(mime) && approxBase64Bytes(data) <= MAX_INLINE_BYTES) {
          parts.push({ inline_data: { mime_type: mime, data } });
        }
      }
      return { role: safeRole(m.role), parts };
    })
    .map(limitParts)
    .filter(m => m.parts && m.parts.length);
}
function limitParts(msg) {
  let used = 0;
  const parts = [];
  for (const p of msg.parts) {
    if (p.text) {
      const t = String(p.text);
      const chunk = t.slice(0, Math.max(0, MAX_TEXT_CHARS - used));
      if (chunk) { parts.push({ text: chunk }); used += chunk.length; }
      if (used >= MAX_TEXT_CHARS) break;
    } else {
      parts.push(p);
    }
  }
  return { ...msg, parts };
}
function coerceData(obj) {
  if (!obj) return { mime: "", data: "" };
  if (typeof obj === "string" && obj.startsWith("data:")) return fromDataUrl(obj);
  if (obj && typeof obj === "object") {
    const mime = obj.mime || obj.mime_type || "";
    const data = obj.data || obj.base64 || (obj.dataUrl ? fromDataUrl(obj.dataUrl).data : "");
    return { mime, data };
  }
  return { mime: "", data: "" };
}
function fromDataUrl(dataUrl) {
  const comma = dataUrl.indexOf(",");
  const header = dataUrl.slice(5, comma);
  const mime = header.includes(";") ? header.slice(0, header.indexOf(";")) : header;
  const data = dataUrl.slice(comma + 1);
  return { mime, data };
}
function approxBase64Bytes(b64) {
  const len = b64.length - (b64.endsWith("==") ? 2 : b64.endsWith("=") ? 1 : 0);
  return Math.floor(len * 0.75);
}

/* ---------- Generation tuning ---------- */
function tuneGeneration(mode, expect = "") {
  const wantsJson = expect === "json";
  if (mode === "qa" || wantsJson) {
    return { temperature: 0.22, topP: 0.9, maxOutputTokens: 4096, candidateCount: 1, responseMimeType: "application/json" };
  }
  if (mode === "plan") {
    return { temperature: 0.28, topP: 0.88, maxOutputTokens: 6144, candidateCount: 1, responseMimeType: "application/json" };
  }
  if (mode === "image_brief") {
    return { temperature: 0.25, topP: 0.85, maxOutputTokens: 1536, candidateCount: 1, responseMimeType: "text/plain" };
  }
  return { temperature: 0.6, topP: 0.9, maxOutputTokens: 4096, candidateCount: 1, responseMimeType: "text/plain" };
}

/* ---------- Safety ---------- */
function buildSafety(level = "strict") {
  const cat = (name) => ({ category: name, threshold: level === "relaxed" ? "BLOCK_NONE" : "BLOCK_ONLY_HIGH" });
  return [
    cat("HARM_CATEGORY_HARASSMENT"),
    cat("HARM_CATEGORY_HATE_SPEECH"),
    cat("HARM_CATEGORY_SEXUALLY_EXPLICIT"),
    cat("HARM_CATEGORY_DANGEROUS_CONTENT"),
  ];
}

/* ---------- Auto-continue helpers ---------- */
function continuePrompt(lang) {
  return (lang === "ar")
    ? "تابع من حيث توقفت بنفس اللغة والبنية، بدون تكرار أو تلخيص؛ أكمل مباشرة."
    : "Continue exactly where you stopped, same language/structure, no repetition or summary; output only the continuation.";
}
function shouldContinue(text) {
  if (!text) return false;
  const tail = text.slice(-80).trim();
  return /[……]$/.test(tail) || /(?:continued|to be continued)[:.]?$/i.test(tail) || tail.endsWith("-") || /\bcontinue\b[:.]?$/i.test(tail);
}
function dedupeContinuation(prev, next) {
  if (!next) return "";
  const head = next.slice(0, 200);
  if (prev && prev.endsWith(head)) return next.slice(head.length).trimStart();
  return next;
}

/* ---------- JSON enforcement/finalization ---------- */
function finalize(text, lang, expect, mode) {
  const expectingJson = (expect === "json" || mode === "plan" || mode === "qa");
  if (expectingJson) {
    const repaired = tryExtractJson(text);
    if (repaired.ok) return JSON.stringify(repaired.json);
    return text;
  }
  return mirrorLanguage(text, lang);
}
function tryExtractJson(s) {
  if (!s) return { ok: false };
  let raw = s.replace(/```json|```/g, "").trim();
  const first = raw.indexOf("{"); const last = raw.lastIndexOf("}");
  if (first >= 0 && last > first) raw = raw.slice(first, last + 1);
  try { const j = JSON.parse(raw); return { ok: true, json: j }; }
  catch { return { ok: false }; }
}

/* ---------- Network & retry ---------- */
async function tryStreamOnce(url, body, timeout, headersForError) {
  for (let attempt = 1; attempt <= MAX_TRIES; attempt++) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeout);
    try {
      const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body, signal: ctrl.signal });
      clearTimeout(t);
      if (!response.ok) {
        if (shouldRetry(response.status) && attempt < MAX_TRIES) { await sleep(attempt); continue; }
        const text = await response.text();
        const data = safeParse(text);
        return { ok: false, errorResp: respond(mapStatus(response.status), headersForError, collectUpstream(response.status, data, text, false)) };
      }
      return { ok: true, response };
    } catch (e) {
      clearTimeout(t);
      if (attempt < MAX_TRIES) { await sleep(attempt); continue; }
      return { ok: false, errorResp: respond(500, headersForError, { error: "Network/timeout", details: String(e && e.message || e) }) };
    }
  }
}

async function tryJSONOnce(url, body, timeout, includeRaw) {
  for (let attempt = 1; attempt <= MAX_TRIES; attempt++) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeout);
    try {
      const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body, signal: ctrl.signal });
      clearTimeout(t);

      const text = await r.text();
      let data; try { data = JSON.parse(text); } catch { data = null; }

      if (!r.ok) {
        if (shouldRetry(r.status) && attempt < MAX_TRIES) { await sleep(attempt); continue; }
        return { ok: false, statusCode: mapStatus(r.status), error: collectUpstream(r.status, data, text, includeRaw) };
      }

      const parts = data?.candidates?.[0]?.content?.parts || [];
      const out = parts.map(p => p?.text || "").join("\n").trim();

      if (!out) {
        const safety = data?.promptFeedback || data?.candidates?.[0]?.safetyRatings;
        return { ok: false, statusCode: 502, error: { error: "Empty/blocked response", safety, raw: includeRaw ? data : undefined } };
      }

      const usage = data?.usageMetadata ? {
        promptTokenCount: data.usageMetadata.promptTokenCount,
        candidatesTokenCount: data.usageMetadata.candidatesTokenCount,
        totalTokenCount: data.usageMetadata.totalTokenCount
      } : undefined;

      return { ok: true, text: out, usage };
    } catch (e) {
      clearTimeout(t);
      if (attempt < MAX_TRIES) { await sleep(attempt); continue; }
      return { ok: false, statusCode: 500, error: { error: "Network/timeout", details: String(e && e.message || e) } };
    }
  }
}

function collectUpstream(status, data, rawText, includeRaw) {
  return {
    error: "Upstream rejected",
    status,
    message: (data && (data.error?.message || data.message)) || String(rawText).slice(0, 1000),
    raw: includeRaw ? data : undefined
  };
}
function shouldRetry(status) { return status === 429 || (status >= 500 && status <= 599); }
function mapStatus(s) { if (s === 429) return 429; if (s >= 500) return 502; return s || 500; }
async function sleep(attempt) { const base = BASE_BACKOFF_MS * Math.pow(2, attempt - 1); const jitter = Math.floor(Math.random() * 400); await new Promise(r => setTimeout(r, base + jitter)); }
function safeParse(s) { try { return JSON.parse(s); } catch { return null; } }
