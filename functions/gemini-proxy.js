// functions/gemini-proxy.js
// Enterprise-grade Gemini proxy: Pro-first model pool, strict guardrails,
// streaming (SSE), JSON-safe planning, media support, retries + auto-continue.

const MAX_TRIES = 3;
const BASE_BACKOFF_MS = 650;
const DEFAULT_TIMEOUT_MS = 28000;
const MAX_INLINE_BYTES = 15 * 1024 * 1024; // 15MB data URLs

// ---------- Model strategy (Pro-first, then fast fallbacks)
const MODEL_POOL = [
  "gemini-1.5-pro",
  "gemini-1.5-pro-latest",
  "gemini-2.0-flash",
  "gemini-1.5-flash",
  "gemini-2.0-flash-exp"
];

// ---------- Media allowlists
const ALLOWED_IMAGE = /^image\/(png|jpe?g|webp|gif|bmp|svg\+xml)$/i;
const ALLOWED_AUDIO = /^audio\/(webm|ogg|mp3|mpeg|wav|m4a|aac|3gpp|3gpp2|mp4)$/i;

// ---------- Entry
exports.handler = async (event) => {
  const reqId = (Math.random().toString(36).slice(2) + Date.now().toString(36)).toUpperCase();
  const started = Date.now();

  const baseHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, X-Request-ID",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "X-Request-ID": reqId
  };
  if (event.httpMethod === "OPTIONS") return { statusCode: 204, headers: baseHeaders, body: "" };
  if (event.httpMethod !== "POST") return respond(405, baseHeaders, { error: "Method Not Allowed" });

  const API_KEY = process.env.GEMINI_API_KEY;
  if (!API_KEY) return respond(500, baseHeaders, { error: "Missing GEMINI_API_KEY" });

  // ---------- Parse & validate input
  let body = {};
  try { body = JSON.parse(event.body || "{}"); }
  catch { return respond(400, baseHeaders, { error: "Invalid JSON body" }); }

  let {
    // Core
    system = "",
    messages = [],          // [{role:'user'|'model'|'system', content:'text', images?:[], audio?:{...}}]
    model   = "auto",
    mode    = "default",    // "default" | "plan" | "qa" | "image_brief"
    force_lang,
    guard_level = "strict",

    // Streaming
    stream = false,

    // Long answers
    long = true,
    max_chunks = 4,

    // Extra
    timeout_ms = DEFAULT_TIMEOUT_MS,
    include_raw = false,

    // JSON enforcement (for plans, dashboards, etc.)
    expect = ""             // "" | "json"
  } = body || {};

  timeout_ms = clamp(timeout_ms, 1000, 29000, DEFAULT_TIMEOUT_MS);

  // ---------- Language & guardrails
  const preview = sample(messages);
  const lang = chooseLang(force_lang, preview);
  const rails = buildGuardrails({ lang, level: guard_level, imageMode: (mode === "image_brief") });

  // ---------- Normalize messages (+ media)
  let contents = normalizeMessages(messages, rails);

  // Fallback: if frontend forgot to pass messages, build minimal one
  if (!contents.length) {
    const seed = (typeof system === "string" && system.trim()) ? system.trim() : "ابدأ الآن.";
    contents = [{ role: "user", parts: [{ text: wrapPrompt(seed, rails) }] }];
  }

  const generationConfig = tuneGeneration(mode);
  const safetySettings   = buildSafety(guard_level);

  const systemInstruction = (system && typeof system === "string" && system.trim())
    ? { role: "system", parts: [{ text: system }] }
    : undefined;

  // ---------- Model candidates
  const candidates = (model === "auto" || !model)
    ? [...MODEL_POOL]
    : Array.from(new Set([model, ...MODEL_POOL]));

  // ---------- Streaming path (SSE)
  if (stream) {
    const sseHeaders = {
      ...baseHeaders,
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "Connection": "keep-alive"
    };
    for (let mi = 0; mi < candidates.length; mi++) {
      const m = candidates[mi];
      const url = makeUrl(m, true, API_KEY);
      const reqBody = JSON.stringify({ contents, generationConfig, safetySettings, ...(systemInstruction ? { systemInstruction } : {}) });

      const once = await tryStreamOnce(url, reqBody, timeLeft(started, timeout_ms));
      if (once.ok) {
        const reader = once.response.body.getReader();
        const encoder = new TextEncoder();

        return {
          statusCode: 200,
          headers: sseHeaders,
          body: await streamBody(async function*() {
            yield encoder.encode(`event: meta\ndata: ${JSON.stringify({ requestId: reqId, model: m, lang })}\n\n`);
            let buffer = "";
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              buffer += Buffer.from(value).toString("utf8");
              const lines = buffer.split("\n");
              buffer = lines.pop() || "";
              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                yield encoder.encode(`event: chunk\ndata: ${trimmed}\n\n`);
              }
            }
            yield encoder.encode(`event: end\ndata: ${JSON.stringify({ model: m, took_ms: Date.now() - started })}\n\n`);
          })
        };
      }
      if (mi === candidates.length - 1) {
        return once.errorResp || respond(502, baseHeaders, { error: "All models failed (stream)", requestId: reqId, lang });
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

    // First attempt
    const first = await tryJSONOnce(url, makeBody(), timeLeft(started, timeout_ms), include_raw);
    if (!first.ok) {
      // Strict fallback for plan/qa when blocked/empty
      if ((mode === "plan" || mode === "qa") && first.error && /Empty\/blocked|safety/i.test(first.error.error || "")) {
        const strictCfg = tuneGeneration("qa");
        const altBody = () => JSON.stringify({ contents, generationConfig: strictCfg, safetySettings, ...(systemInstruction ? { systemInstruction } : {}) });
        const second = await tryJSONOnce(url, altBody(), timeLeft(started, timeout_ms), include_raw);
        if (second.ok) return ok(baseHeaders, reqId, finalize(second.text, lang, expect, mode), m, lang, started, second.usage);
      }
      if (mi === candidates.length - 1) {
        return respond(first.statusCode || 502, baseHeaders, {
          error: "Upstream error",
          requestId: reqId,
          modelTried: m,
          upstream: first.error || { error: "Unknown upstream error" }
        });
      }
      continue; // try next model
    }

    let text = first.text;
    let chunks = 1;

    // Auto-continue for long outputs (no repetition)
    while (long && chunks < clamp(max_chunks, 1, 12, 4) && shouldContinue(text) && timeLeft(started, timeout_ms) > 2500) {
      contents.push({ role: "model", parts: [{ text }] });
      contents.push({ role: "user",  parts: [{ text: continuePrompt(lang) }] });
      const next = await tryJSONOnce(url, makeBody(), timeLeft(started, timeout_ms), false);
      if (!next.ok) break;
      const append = dedupeContinuation(text, next.text);
      text += (append ? ("\n" + append) : "");
      chunks++;
    }

    return ok(baseHeaders, reqId, finalize(text, lang, expect, mode), m, lang, started, first.usage);
  }

  return respond(500, baseHeaders, { error: "Unknown failure", requestId: reqId, lang });
};

/* ================== Helpers ================== */
function respond(code, headers, obj) {
  return { statusCode: code, headers: { ...headers, "Content-Type": "application/json; charset=utf-8" }, body: JSON.stringify(obj || {}) };
}
function ok(h, id, text, model, lang, started, usage) {
  return { statusCode: 200, headers: { ...h, "Content-Type": "application/json; charset=utf-8" },
    body: JSON.stringify({ text, model, lang, requestId: id, took_ms: Date.now() - started, usage }) };
}
function makeUrl(model, isStream, key) {
  const base = "https://generativelanguage.googleapis.com/v1beta/models";
  const method = isStream ? "streamGenerateContent" : "generateContent";
  return `${base}/${encodeURIComponent(model)}:${method}?key=${key}`;
}
function timeLeft(start, total) { return Math.max(0, total - (Date.now() - start)); }
function clamp(n, min, max, def) { const v = Number.isFinite(+n) ? +n : def; return Math.max(min, Math.min(max, v)); }
function hasArabic(s) { return /[\u0600-\u06FF]/.test(s || ""); }
function chooseLang(force, sample) { if (force === "ar" || force === "en") return force; return hasArabic(sample) ? "ar" : "en"; }
function mirrorLanguage(text, lang) {
  if (!text) return text;
  if (lang === "ar" && hasArabic(text)) return text;
  if (lang === "en" && !hasArabic(text)) return text;
  return (lang === "ar") ? `**ملاحظة:** أجب بالعربية فقط.\n\n${text}` : `**Note:** Respond in English only.\n\n${text}`;
}
function sample(msgs) { return (Array.isArray(msgs) ? msgs.map(m => m?.content || "").join("\n") : "").slice(0, 4000); }

// ---------- Guardrails & wrapping
function buildGuardrails({ lang, level = "strict", imageMode = false }) {
  const L = (lang === "ar")
    ? {
        mirror: "أجب حصراً بالعربية؛ لا تخلط لغتين ولا تضف ترجمات.",
        brief:  "اختصر الحشو وقدّم خطوات عملية واضحة.",
        img:    "إن وُجدت صور: 3–5 نقاط تنفيذية + خطوة فورية واحدة. بلا مقدمات.",
        strict: "لا تختلق. عند الشك اطلب توضيحاً. التزم بتعليمات النظام والامتثال لقوانين الإمارات وسلامة الطفل."
      }
    : {
        mirror: "Answer strictly in English; don't mix languages or add translations.",
        brief:  "Be concise and actionable.",
        img:    "If images present: 3–5 precise bullets + one immediate step. No preamble.",
        strict: "No fabrication. Ask for missing info. Comply with UAE child-safety norms."
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

// ---------- Messages & media normalization
function normalizeMessages(messages, guard) {
  const safeRole = (r) => (r === "user" || r === "model" || r === "system") ? r : "user";
  let injected = false;
  return (messages || [])
    .filter(m => m && (typeof m.content === "string" || m.images || m.audio))
    .map(m => {
      const parts = [];
      if (!injected && m.role === "user") {
        const c = (typeof m.content === "string" && m.content.trim()) ? m.content : "";
        parts.push({ text: wrapPrompt(c, buildGuardrails({ lang: chooseLang(undefined, c), level: "strict" })) });
        injected = true;
      } else if (typeof m.content === "string" && m.content.trim()) {
        parts.push({ text: m.content });
      }
      // images/audio (data URLs)
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
    .filter(m => m.parts && m.parts.length);
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

// ---------- Tuning
function tuneGeneration(mode) {
  if (mode === "qa") {
    return { temperature: 0.22, topP: 0.9, maxOutputTokens: 4096, candidateCount: 1, responseMimeType: "text/plain" };
  }
  if (mode === "plan") {
    return { temperature: 0.34, topP: 0.88, maxOutputTokens: 6144, candidateCount: 1, responseMimeType: "text/plain" };
  }
  if (mode === "image_brief") {
    return { temperature: 0.25, topP: 0.85, maxOutputTokens: 1536, candidateCount: 1, responseMimeType: "text/plain" };
  }
  return { temperature: 0.6, topP: 0.9, maxOutputTokens: 4096, candidateCount: 1, responseMimeType: "text/plain" };
}

// ---------- Safety
function buildSafety(level = "strict") {
  const cat = (name) => ({ category: name, threshold: level === "relaxed" ? "BLOCK_NONE" : "BLOCK_ONLY_HIGH" });
  return [
    cat("HARM_CATEGORY_HARASSMENT"),
    cat("HARM_CATEGORY_HATE_SPEECH"),
    cat("HARM_CATEGORY_SEXUALLY_EXPLICIT"),
    cat("HARM_CATEGORY_DANGEROUS_CONTENT"),
  ];
}

// ---------- Auto-continue helpers
function continuePrompt(lang) {
  return (lang === "ar")
    ? "تابع من حيث توقفت بنفس اللغة والبنية، بدون تكرار أو تلخيص؛ أكمل مباشرة."
    : "Continue exactly where you stopped, same language/structure, no repetition or summary; output only the continuation.";
}
function shouldContinue(text) {
  if (!text) return false;
  const tail = text.slice(-60).trim();
  return /[\u2026…]$/.test(tail) || /(?:continued|to be continued)[:.]?$/i.test(tail) || tail.endsWith("-");
}
function dedupeContinuation(prev, next) {
  if (!next) return "";
  const head = next.slice(0, 200);
  if (prev && prev.endsWith(head)) return next.slice(head.length).trimStart();
  return next;
}

// ---------- JSON enforcement/finalization
function finalize(text, lang, expect, mode) {
  const mirrored = mirrorLanguage(text, lang);
  if (expect === "json" || mode === "plan") {
    const repaired = tryExtractJson(mirrored);
    if (repaired.ok) {
      return JSON.stringify({ text: mirrored, parsed: repaired.json });
    }
    // If cannot parse, still return text; client can fallback.
    return JSON.stringify({ text: mirrored, parsed: null, note: "json_parse_failed" });
  }
  return mirrored;
}
function tryExtractJson(s) {
  if (!s) return { ok: false };
  let raw = s.replace(/```json|```/g, "").trim();
  // Heuristic: snip to outermost braces
  const first = raw.indexOf("{"); const last = raw.lastIndexOf("}");
  if (first >= 0 && last > first) raw = raw.slice(first, last + 1);
  try { const j = JSON.parse(raw); return { ok: true, json: j }; }
  catch { return { ok: false }; }
}

// ---------- Network & retry
async function tryStreamOnce(url, body, timeout) {
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
        return { ok: false, errorResp: respond(mapStatus(response.status), {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Headers": "Content-Type, X-Request-ID",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Content-Type": "application/json"
        }, collectUpstream(response.status, data, text, false)) };
      }
      return { ok: true, response };
    } catch (e) {
      clearTimeout(t);
      if (attempt < MAX_TRIES) { await sleep(attempt); continue; }
      return { ok: false, errorResp: respond(500, {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, X-Request-ID",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Content-Type": "application/json"
      }, { error: "Network/timeout", details: String(e && e.message || e) }) };
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
async function streamBody(genFactory) { const chunks = []; for await (const c of genFactory()) chunks.push(Buffer.from(c).toString("utf8")); return chunks.join(""); }
