// functions/gemini-proxy.js
// Pro-first Gemini proxy with rich error details + smart fallback for plan/chat.

const MAX_TRIES = 3;
const BASE_BACKOFF_MS = 600;
const DEFAULT_TIMEOUT_MS = 28000;
const MAX_INLINE_BYTES = 15 * 1024 * 1024;

const MODEL_POOL = [
  "gemini-1.5-pro",
  "gemini-1.5-pro-latest",
  "gemini-2.0-flash",
  "gemini-1.5-flash",
  "gemini-2.0-flash-exp"
];

exports.handler = async (event) => {
  const reqId = (Math.random().toString(36).slice(2)+Date.now().toString(36)).toUpperCase();
  const start = Date.now();
  const baseHeaders = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, X-Request-ID",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "X-Request-ID": reqId
  };
  if (event.httpMethod === "OPTIONS") return { statusCode: 204, headers: baseHeaders, body: "" };
  if (event.httpMethod !== "POST") return resp(405, baseHeaders, { error: "Method Not Allowed" });

  const API_KEY = process.env.GEMINI_API_KEY;
  if (!API_KEY) return resp(500, baseHeaders, { error: "Missing GEMINI_API_KEY" });

  let inBody = {};
  try { inBody = JSON.parse(event.body || "{}"); }
  catch { return resp(400, baseHeaders, { error: "Invalid JSON body" }); }

  const {
    system = "",
    messages = [],
    model = "auto",
    mode = "default",           // "default" | "plan" | "image_brief"
    force_lang,
    guard_level = "strict",
    long = true,
    max_chunks = 3,
    timeout_ms = DEFAULT_TIMEOUT_MS
  } = inBody;

  // Build request
  const preview = sampleFrom(messages);
  const lang = chooseLang(force_lang, preview);
  const guard = guardrails(lang, guard_level);

  const contents = normalizeMessages(messages, guard);
  const generationConfig = tune(mode);
  const safetySettings   = buildSafety(guard_level);

  const systemInstruction = system ? { role: "system", parts: [{ text: system }] } : undefined;
  const candidates = (model === "auto" || !model) ? [...MODEL_POOL] : Array.from(new Set([model, ...MODEL_POOL]));

  // Non-stream with fallback + auto-continue
  for (const m of candidates) {
    const url = makeUrl(m, false, API_KEY);
    const makeBody = () => JSON.stringify({ contents, generationConfig, safetySettings, ...(systemInstruction ? { systemInstruction } : {}) });

    const first = await tryJSONOnce(url, makeBody(), timeout_msLeft(start, timeout_ms), true);
    if (!first.ok) {
      // إذا كان حجب/سلامة—جرّب نمط مبسّط (أكثر تحفظاً) للخطة
      if (mode === "plan" && first.error && /Empty\/blocked|safety/i.test(first.error.error || "")) {
        const strictCfg = tune("qa");
        const b = () => JSON.stringify({ contents, generationConfig: strictCfg, safetySettings, ...(systemInstruction ? { systemInstruction } : {}) });
        const second = await tryJSONOnce(url, b(), timeout_msLeft(start, timeout_ms), true);
        if (second.ok) return ok(baseHeaders, reqId, second.text, m, lang, start, second.usage);
        // وإلاّ نعيد الخطأ بتفاصيله
      }
      if (m === candidates[candidates.length - 1]) {
        return resp(first.statusCode || 502, baseHeaders, {
          error: "Upstream error",
          requestId: reqId,
          modelTried: m,
          upstream: first.error || { error: "Unknown upstream error" }
        });
      }
      continue;
    }

    // auto-continue
    let text = first.text, chunks = 1;
    while (long && chunks < Math.max(1, Math.min(8, max_chunks)) && shouldContinue(text) && timeout_msLeft(start, timeout_ms) > 2500) {
      contents.push({ role: "model", parts: [{ text }] });
      contents.push({ role: "user", parts: [{ text: contPrompt(lang) }] });
      const next = await tryJSONOnce(url, makeBody(), timeout_msLeft(start, timeout_ms), false);
      if (!next.ok) break;
      const append = dedupe(text, next.text);
      text += (append ? ("\n" + append) : "");
      chunks++;
    }

    return ok(baseHeaders, reqId, text, m, lang, start, first.usage);
  }

  return resp(500, baseHeaders, { error: "Unknown failure", requestId: reqId });

  /* helpers */
  function ok(h, id, text, model, lang, started, usage) {
    return {
      statusCode: 200,
      headers: { ...h, "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ text: mirrorLang(text, lang), model, lang, requestId: id, took_ms: Date.now()-started, usage })
    };
  }
};

function resp(code, headers, obj){ return { statusCode: code, headers: { ...headers, "Content-Type":"application/json" }, body: JSON.stringify(obj||{}) }; }
function timeout_msLeft(start,total){ return Math.max(0, total - (Date.now()-start)); }
function makeUrl(model, stream, key){ const b="https://generativelanguage.googleapis.com/v1beta/models"; const m=stream?"streamGenerateContent":"generateContent"; return `${b}/${encodeURIComponent(model)}:${m}?key=${key}`; }
function hasArabic(s){ return /[\u0600-\u06FF]/.test(s||""); }
function chooseLang(force,sample){ if(force==="ar"||force==="en") return force; return hasArabic(sample)?"ar":"en"; }
function mirrorLang(t,lang){ if(!t) return t; if(lang==="ar"&&hasArabic(t)) return t; if(lang==="en"&&!hasArabic(t)) return t; return (lang==="ar")?`**ملاحظة:** الرد بالعربية فقط.\n\n${t}`:`**Note:** Response in English only.\n\n${t}`; }
function sampleFrom(msgs){ return (Array.isArray(msgs)?msgs.map(m=>m?.content||"").join("\n"):"").slice(0,2000); }

function guardrails(lang, level){
  const L = (lang==="ar") ? [
    "أجب بالعربية حصراً؛ لا تخلط لغتين.",
    "لا تختلق؛ عند الشك اطلب توضيحاً. التزم بتعليمات السلامة والتشريعات بالإمارات.",
    "اختصر الحشو وقدّم خطوات قابلة للتنفيذ بلغة إنسانية.",
  ] : [
    "Answer strictly in English; do not mix languages.",
    "No fabrication; ask for missing info. Comply with UAE child-safety norms.",
    "Cut fluff; output actionable steps."
  ];
  if (level !== "relaxed") L.push((lang==="ar")?"التزم بالتوجيهات حرفياً.":"Follow the system instructions exactly.");
  return L.join("\n");
}
function normalizeMessages(messages, guard){
  const safeRole = r => (r==="user"||r==="model"||r==="system") ? r : "user";
  let injected=false;
  return (messages||[])
    .filter(m=>m && typeof m.content==="string")
    .map(m=>{
      const parts=[];
      if (!injected && m.role==="user") { parts.push({ text: `تعليمات ملزمة:\n${guard}\n\n---\n${m.content}` }); injected=true; }
      else { parts.push({ text: m.content }); }
      return { role: safeRole(m.role), parts };
    })
    .filter(m=>m.parts.length);
}
function tune(mode){
  if (mode==="qa"){ return { temperature: 0.22, topP: 0.9, maxOutputTokens: 4096, candidateCount: 1, responseMimeType: "text/plain" }; }
  if (mode==="plan"){ return { temperature: 0.35, topP: 0.88, maxOutputTokens: 6144, candidateCount: 1, responseMimeType: "text/plain" }; }
  return { temperature: 0.6, topP: 0.9, maxOutputTokens: 4096, candidateCount: 1, responseMimeType: "text/plain" };
}
function contPrompt(lang){ return (lang==="ar") ? "تابع من حيث انتهيت بنفس اللغة والبناء بدون تكرار." : "Continue from where you stopped, same language/structure, no repetition."; }
function shouldContinue(t){ if(!t) return false; const tail=t.slice(-40).trim(); return /[\u2026…]$/.test(tail)||/(?:continued|to be continued)[:.]?$/i.test(tail)||tail.endsWith("-"); }
function dedupe(prev,next){ if(!next) return ""; const head=next.slice(0,200); if(prev && prev.endsWith(head)) return next.slice(head.length).trimStart(); return next; }

async function tryJSONOnce(url, body, timeout, includeRaw){
  for (let attempt=1; attempt<=MAX_TRIES; attempt++){
    const ctrl = new AbortController(); const t = setTimeout(()=>ctrl.abort(), timeout);
    try{
      const r = await fetch(url,{ method:"POST", headers:{ "Content-Type":"application/json" }, body, signal:ctrl.signal });
      clearTimeout(t);
      const text = await r.text(); let data; try{ data=JSON.parse(text); }catch{ data=null; }
      if (!r.ok){
        if ((r.status===429 || (r.status>=500 && r.status<=599)) && attempt<MAX_TRIES) { await sleep(attempt); continue; }
        return { ok:false, statusCode: mapStatus(r.status), error: collectUpstream(r.status, data, text, includeRaw) };
      }
      const parts = data?.candidates?.[0]?.content?.parts || [];
      const out = parts.map(p=>p?.text||"").join("\n").trim();
      if (!out) {
        const safety = data?.promptFeedback || data?.candidates?.[0]?.safetyRatings;
        return { ok:false, statusCode:502, error:{ error:"Empty/blocked response", safety, raw: includeRaw?data:undefined } };
      }
      const usage = data?.usageMetadata ? {
        promptTokenCount: data.usageMetadata.promptTokenCount,
        candidatesTokenCount: data.usageMetadata.candidatesTokenCount,
        totalTokenCount: data.usageMetadata.totalTokenCount
      } : undefined;
      return { ok:true, text: out, usage };
    }catch(e){
      clearTimeout(t);
      if (attempt<MAX_TRIES){ await sleep(attempt); continue; }
      return { ok:false, statusCode:500, error:{ error:"Network/timeout", details:String(e && e.message || e) } };
    }
  }
}
function collectUpstream(status, data, raw, includeRaw){
  return {
    error: "Upstream rejected",
    status,
    message: (data && (data.error?.message || data.message)) || String(raw).slice(0,1000),
    raw: includeRaw ? data : undefined
  };
}
function mapStatus(s){ if(s===429) return 429; if(s>=500) return 502; return s||500; }
async function sleep(attempt){ const base = BASE_BACKOFF_MS * Math.pow(2, attempt-1); const jitter = Math.floor(Math.random()*400); await new Promise(r=>setTimeout(r, base+jitter)); }
