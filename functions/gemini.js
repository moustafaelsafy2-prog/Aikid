// functions/gemini-proxy.js
// Pro-first, accuracy-tuned Gemini proxy: retries, guardrails, auto-continue, streaming.

const MAX_TRIES = 3;
const BASE_BACKOFF_MS = 600;
const MAX_OUTPUT_TOKENS_HARD = 8192;
const DEFAULT_TIMEOUT_MS = 28000;

const MAX_INLINE_BYTES = 15 * 1024 * 1024;
const ALLOWED_IMAGE = /^image\/(png|jpe?g|webp|gif|bmp|svg\+xml)$/i;
const ALLOWED_AUDIO = /^audio\/(webm|ogg|mp3|mpeg|wav|m4a|aac|3gpp|3gpp2|mp4)$/i;

// دقة أولاً (Pro/Flash)، ويمكن تمرير model="auto"
const MODEL_POOL = [
  "gemini-1.5-pro",
  "gemini-1.5-pro-latest",
  "gemini-2.0-flash",
  "gemini-1.5-flash",
  "gemini-2.0-flash-exp"
];

exports.handler = async (event) => {
  const requestId = (Math.random().toString(36).slice(2) + Date.now().toString(36)).toUpperCase();
  const startedAt = Date.now();

  const baseHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, X-Request-ID",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "X-Request-ID": requestId
  };
  if (event.httpMethod === "OPTIONS") return { statusCode: 204, headers: baseHeaders, body: "" };
  if (event.httpMethod !== "POST") return resp(405, baseHeaders, { error: "Method Not Allowed" });

  const API_KEY = process.env.GEMINI_API_KEY;
  if (!API_KEY) return resp(500, baseHeaders, { error: "GEMINI_API_KEY is missing" });

  let payload;
  try { payload = JSON.parse(event.body || "{}"); }
  catch { return resp(400, baseHeaders, { error: "Invalid JSON" }); }

  // واجهة الاستدعاء
  let {
    prompt,                 // نص حر (إن لم تُرسل messages)
    messages,               // [{role:"user"|"model"|"system", content:"..." , images?:[], audio?:{...}}]
    images,
    audio,
    model = "auto",
    temperature,
    top_p,
    max_output_tokens,
    system,                 // system prompt
    stream = false,
    timeout_ms = DEFAULT_TIMEOUT_MS,
    include_raw = false,

    // إعدادات ذكية إضافية
    mode = "default",       // "default" | "qa" | "image_brief"
    force_lang,             // "ar" | "en"
    concise_image,          // boolean
    guard_level = "strict", // "relaxed" | "strict"

    // تكملة تلقائية للنص الطويل
    long = true,
    max_chunks = 4
  } = payload || {};

  timeout_ms = clampNumber(timeout_ms, 1000, 29000, DEFAULT_TIMEOUT_MS);

  const preview = textPreview(prompt || messages?.map(m => m?.content || "").join("\n"));
  const lang = chooseLang(force_lang, preview);
  const hasAnyImages = !!(images?.length) || !!(Array.isArray(messages) && messages.some(m => m.images?.length));
  const useImageBrief = !!(concise_image || mode === "image_brief" || hasAnyImages);

  const guard = buildGuardrails({ lang, useImageBrief, level: guard_level });

  // حوّل الرسائل لصيغة Gemini
  const contents = Array.isArray(messages)
    ? normalizeMessagesWithMedia(messages, guard)
    : [{ role: "user", parts: buildParts(wrapPrompt(prompt, lang, useImageBrief, guard), images, audio) }];

  const generationConfig = tuneGeneration({ temperature, top_p, max_output_tokens, useImageBrief, mode });
  const safetySettings   = buildSafety(guard_level);

  const systemInstruction = (system && typeof system === "string")
    ? { role: "system", parts: [{ text: system }] }
    : undefined;

  // قائمة النماذج المرشحة
  const candidates = (model === "auto" || !model)
    ? [...MODEL_POOL]
    : Array.from(new Set([model, ...MODEL_POOL]));

  // ===== Streaming (SSE) =====
  if (stream) {
    const headers = { ...baseHeaders, "Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache, no-transform", "Connection": "keep-alive" };
    for (const m of candidates) {
      const url = makeUrl(m, true, API_KEY);
      const body = JSON.stringify({ contents, generationConfig, safetySettings, ...(systemInstruction ? { systemInstruction } : {}) });

      const sseOnce = await tryStreamOnce(url, body, timeout_ms);
      if (sseOnce.ok) {
        const reader = sseOnce.response.body.getReader();
        const encoder = new TextEncoder();
        return {
          statusCode: 200,
          headers,
          body: await streamBody(async function* () {
            yield encoder.encode(`event: meta\ndata: ${JSON.stringify({ requestId, model: m, lang })}\n\n`);
            let buffer = "";
            while (true) {
              const { value, done } = await reader.read();
              if (done) break;
              buffer += Buffer.from(value).toString("utf8");
              const lines = buffer.split("\n"); buffer = lines.pop() || "";
              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                yield encoder.encode(`event: chunk\ndata: ${trimmed}\n\n`);
              }
            }
            yield encoder.encode(`event: end\ndata: ${JSON.stringify({ model: m, took_ms: Date.now() - startedAt })}\n\n`);
          })
        };
      }
      // آخر محاولة فاشلة
      if (m === candidates[candidates.length - 1]) return resp(502, baseHeaders, { error: "All models failed (stream)", requestId, lang });
    }
  }

  // ===== Non-stream + Fallback + Auto-continue =====
  for (const m of candidates) {
    const url = makeUrl(m, false, API_KEY);
    const makeBody = () => JSON.stringify({ contents, generationConfig, safetySettings, ...(systemInstruction ? { systemInstruction } : {}) });

    const first = await tryJSONOnce(url, makeBody(), timeout_msLeft(startedAt, timeout_ms), include_raw);
    if (!first.ok) {
      if (m === candidates[candidates.length - 1]) {
        return resp(first.statusCode || 502, baseHeaders, { ...(first.error || { error: "All models failed" }), requestId, lang });
      }
      continue;
    }

    let text = first.text;
    let chunks = 1;

    while (long && chunks < clampNumber(max_chunks, 1, 12, 4) && shouldContinue(text) && timeout_msLeft(startedAt, timeout_ms) > 2500) {
      contents.push({ role: "model", parts: [{ text }] });
      contents.push({ role: "user",  parts: [{ text: continuePrompt(lang) }] });

      const next = await tryJSONOnce(url, makeBody(), timeout_msLeft(startedAt, timeout_ms), false);
      if (!next.ok) break;

      const append = dedupeContinuation(text, next.text);
      text += (append ? ("\n" + append) : "");
      chunks++;
    }

    return {
      statusCode: 200,
      headers: { ...baseHeaders, "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({
        text: mirrorLanguage(text, lang),
        model: m,
        lang,
        usage: first.usage || undefined,
        requestId,
        took_ms: Date.now() - startedAt
      })
    };
  }

  return resp(500, baseHeaders, { error: "Unknown failure", requestId, lang });
};

/* =================== Helpers =================== */
function resp(statusCode, headers, obj){ return { statusCode, headers, body: JSON.stringify(obj ?? {}) }; }
function clampNumber(n,min,max,fallback){ const v = Number.isFinite(+n)?+n:fallback; return Math.max(min, Math.min(max, v)); }
function makeUrl(model, isStream, apiKey){ const base="https://generativelanguage.googleapis.com/v1beta/models"; const method=isStream?"streamGenerateContent":"generateContent"; return `${base}/${encodeURIComponent(model)}:${method}?key=${apiKey}`; }
function hasArabic(s){ return /[\u0600-\u06FF]/.test(s||""); }
function chooseLang(force,sample){ if(force==="ar"||force==="en") return force; return hasArabic(sample)?"ar":"en"; }
function mirrorLanguage(text,lang){ if(!text) return text; if(lang==="ar"&&hasArabic(text)) return text; if(lang==="en"&&!hasArabic(text)) return text; return (lang==="ar")?`**ملاحظة:** الرد بالعربية فقط.\n\n${text}`:`**Note:** Response in English only.\n\n${text}`; }
function textPreview(s){ return (s||"").slice(0,6000); }

function buildGuardrails({ lang, useImageBrief, level }){
  const L = (lang==="ar") ? {
    mirror:"أجب حصراً باللغة العربية الظاهرة. لا تمزج لغتين.",
    beBrief:"اختصر الحشو وركّز على خطوات قابلة للتنفيذ.",
    imageBrief:"عند الصور: 3–5 نقاط تنفيذية دقيقة + خطوة فورية. بدون مقدمات.",
    strict:"لا تختلق. عند الشك اطلب التوضيح. اتبع التعليمات حرفيًا."
  } : {
    mirror:"Answer strictly in English. Do not mix languages.",
    beBrief:"Cut fluff; output precise, executable steps.",
    imageBrief:"If images: 3–5 actionable bullets + one immediate step. No preamble.",
    strict:"Never fabricate. Ask for missing info. Follow instructions exactly."
  };
  return [L.mirror, L.beBrief, (useImageBrief?L.imageBrief:""), (level!=="relaxed"?L.strict:"")].filter(Boolean).join("\n");
}
function wrapPrompt(prompt,lang,useImageBrief,guard){
  const head = (lang==="ar")?"تعليمات ملزمة:":"Mandatory guardrails:";
  return `${head}\n${guard}\n\n---\n${prompt||""}`;
}
function buildParts(prompt, images, audio){
  const parts=[]; if(typeof prompt==="string" && prompt.trim()) parts.push({ text: prompt });
  parts.push(...coerceMediaParts(images,audio)); return parts;
}
function normalizeMessagesWithMedia(messages, guard){
  const safeRole = r => (r==="user"||r==="model"||r==="system")?r:"user";
  let injected=false;
  return messages
    .filter(m => m && (typeof m.content==="string" || m.images || m.audio))
    .map(m=>{
      const parts=[];
      if(!injected && m.role==="user"){
        const c=(typeof m.content==="string" && m.content.trim())?m.content:"";
        parts.push({ text: wrapPrompt(c, chooseLang(undefined,c), !!(m.images && m.images.length), guard) });
        injected=true;
      } else if(typeof m.content==="string" && m.content.trim()){
        parts.push({ text: m.content });
      }
      parts.push(...coerceMediaParts(m.images, m.audio));
      return { role: safeRole(m.role), parts };
    })
    .filter(m => m.parts.length);
}
function coerceMediaParts(images, audio){
  const parts=[];
  if(Array.isArray(images)){
    for(const item of images){
      let mime,b64;
      if(typeof item==="string" && item.startsWith("data:")){ ({mime,data:b64}=fromDataUrl(item)); }
      else if(item && typeof item==="object"){ mime=item.mime||item.mime_type; b64=item.data||item.base64||(item.dataUrl?fromDataUrl(item.dataUrl).data:""); }
      if(!mime||!b64) continue;
      if(!ALLOWED_IMAGE.test(mime)) continue;
      if(approxBase64Bytes(b64)>MAX_INLINE_BYTES) continue;
      parts.push({ inline_data:{ mime_type: mime, data: b64 } });
    }
  }
  if(audio){
    let mime,b64;
    if(typeof audio==="string" && audio.startsWith("data:")){ ({mime,data:b64}=fromDataUrl(audio)); }
    else if(typeof audio==="object"){ mime=audio.mime||audio.mime_type; b64=audio.data||audio.base64||(audio.dataUrl?fromDataUrl(audio.dataUrl).data:""); }
    if(mime && b64 && ALLOWED_AUDIO.test(mime) && approxBase64Bytes(b64)<=MAX_INLINE_BYTES){
      parts.push({ inline_data:{ mime_type: mime, data: b64 } });
    }
  }
  return parts;
}
function fromDataUrl(dataUrl){ const comma=dataUrl.indexOf(','); const header=dataUrl.slice(5,comma); const mime=header.includes(';')?header.slice(0,header.indexOf(';')):header; const data=dataUrl.slice(comma+1); return { mime, data }; }
function approxBase64Bytes(b64){ const len=b64.length-(b64.endsWith('==')?2:b64.endsWith('=')?1:0); return Math.floor(len*0.75); }

function tuneGeneration({ temperature, top_p, max_output_tokens, useImageBrief, mode }){
  let t   = (temperature==null) ? (useImageBrief?0.25:0.30) : temperature;
  let tp  = (top_p==null) ? 0.88 : top_p;
  let mot = (max_output_tokens==null) ? (useImageBrief?1536:6144) : max_output_tokens;
  if (mode === "qa" || mode === "factual"){ t=Math.min(t,0.24); tp=Math.min(tp,0.9); mot=Math.max(mot,3072); }
  t   = clampNumber(t, 0.0, 1.0, 0.30);
  tp  = clampNumber(tp, 0.0, 1.0, 0.88);
  mot = clampNumber(mot, 1, MAX_OUTPUT_TOKENS_HARD, 6144);
  return { temperature:t, topP:tp, maxOutputTokens:mot, candidateCount:1, responseMimeType:"text/plain" };
}
function buildSafety(level="strict"){
  const cat = (c)=>({ category:c, threshold: level==="relaxed"?"BLOCK_NONE":"BLOCK_ONLY_HIGH" });
  return [
    cat("HARM_CATEGORY_HARASSMENT"),
    cat("HARM_CATEGORY_HATE_SPEECH"),
    cat("HARM_CATEGORY_SEXUALLY_EXPLICIT"),
    cat("HARM_CATEGORY_DANGEROUS_CONTENT")
  ];
}
function continuePrompt(lang){
  return (lang==="ar")
    ? "تابع من حيث توقفت بنفس اللغة والهيكل، بدون تكرار أو تلخيص، واصل مباشرة."
    : "Continue exactly where you stopped, same language and structure, no repetition or summary; only the continuation.";
}
function shouldContinue(text){ if(!text) return false; const tail=text.slice(-40).trim(); return /[\u2026…]$/.test(tail) || /(?:continued|to be continued)[:.]?$/i.test(tail) || tail.endsWith("-"); }
function dedupeContinuation(prev,next){ if(!next) return ""; const head=next.slice(0,200); if(prev && prev.endsWith(head)) return next.slice(head.length).trimStart(); return next; }
function timeout_msLeft(start,total){ return Math.max(0, total - (Date.now()-start)); }

function shouldRetry(status){ return status===429 || (status>=500 && status<=599); }
function mapStatus(s){ if(s===429) return 429; if(s>=500) return 502; return s||500; }
function collectUpstreamError(status,data,text){ const details=(data && (data.error?.message||data.message)) || (typeof text==="string"?text.slice(0,1000):"Upstream error"); return { error:"Upstream error", status, details }; }
async function sleepWithJitter(attempt){ const base=BASE_BACKOFF_MS*Math.pow(2,attempt-1); const jitter=Math.floor(Math.random()*400); await new Promise(r=>setTimeout(r, base + jitter)); }

async function tryStreamOnce(url, body, timeout){
  for(let attempt=1; attempt<=MAX_TRIES; attempt++){
    const abort=new AbortController(); const t=setTimeout(()=>abort.abort(), timeout);
    try{
      const response = await fetch(url,{ method:"POST", headers:{ "Content-Type":"application/json" }, body, signal:abort.signal });
      clearTimeout(t);
      if(!response.ok){
        if(shouldRetry(response.status) && attempt<MAX_TRIES){ await sleepWithJitter(attempt); continue; }
        return { ok:false };
      }
      return { ok:true, response };
    }catch(e){
      clearTimeout(t);
      if(attempt<MAX_TRIES){ await sleepWithJitter(attempt); continue; }
      return { ok:false };
    }
  }
}
async function tryJSONOnce(url, body, timeout, include_raw){
  for(let attempt=1; attempt<=MAX_TRIES; attempt++){
    const abort=new AbortController(); const t=setTimeout(()=>abort.abort(), timeout);
    try{
      const r = await fetch(url,{ method:"POST", headers:{ "Content-Type":"application/json" }, body, signal:abort.signal });
      clearTimeout(t);
      const text = await r.text(); let data; try{ data=JSON.parse(text); }catch{ data=null; }
      if(!r.ok){
        if(shouldRetry(r.status) && attempt<MAX_TRIES){ await sleepWithJitter(attempt); continue; }
        return { ok:false, statusCode: mapStatus(r.status), error: collectUpstreamError(r.status, data, text) };
      }
      const parts = data?.candidates?.[0]?.content?.parts || [];
      const out = parts.map(p=>p?.text||"").join("\n").trim();
      if(!out){
        const safety = data?.promptFeedback || data?.candidates?.[0]?.safetyRatings;
        return { ok:false, statusCode:502, error:{ error:"Empty/blocked response", safety, raw: include_raw?data:undefined } };
      }
      const usage = data?.usageMetadata ? {
        promptTokenCount: data.usageMetadata.promptTokenCount,
        candidatesTokenCount: data.usageMetadata.candidatesTokenCount,
        totalTokenCount: data.usageMetadata.totalTokenCount
      } : undefined;
      return { ok:true, text: out, usage, raw: include_raw?data:undefined };
    }catch(e){
      clearTimeout(t);
      if(attempt<MAX_TRIES){ await sleepWithJitter(attempt); continue; }
      return { ok:false, statusCode:500, error:{ error:"Network/timeout", details:String(e && e.message || e) } };
    }
  }
}
async function streamBody(gen){ const chunks=[]; for await (const buf of gen()) chunks.push(Buffer.from(buf).toString("utf8")); return chunks.join(""); }
