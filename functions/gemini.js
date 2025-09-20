// functions/gemini.js
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const CUSTOM_MODEL   = process.env.GEMINI_MODEL; // (اختياري)
const BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

const PIPE = [
  'gemini-1.5-pro',
  'gemini-1.5-flash'
];
if (CUSTOM_MODEL && !PIPE.includes(CUSTOM_MODEL)) PIPE.unshift(CUSTOM_MODEL);

const ok = (body, status = 200) => ({
  statusCode: status,
  headers: {
    'Content-Type': 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS'
  },
  body: JSON.stringify(body)
});

exports.handler = async (event) => {
  if (event.httpMethod === 'OPTIONS') return ok({});
  if (event.httpMethod !== 'POST') return ok({ error: 'Method not allowed' }, 405);
  if (!GEMINI_API_KEY) return ok({ error: 'Missing GEMINI_API_KEY' }, 500);

  try {
    const { systemInstruction = '', history = [], mode = 'chat' } = JSON.parse(event.body || '{}');

    const contents = Array.isArray(history) ? history.map(m => ({
      role: m.role === 'user' ? 'user' : 'model',
      parts: (m.parts || []).map(p => ({ text: String(p.text || '').slice(0, 8000) }))
    })) : [];

    const baseCfg = mode === 'plan'
      ? { temperature: 0.35, topK: 50, topP: 0.9, maxOutputTokens: 1600 }
      : { temperature: 0.6,  topK: 40, topP: 0.9, maxOutputTokens: 1400 };

    const requestFor = (model) => ({
      contents,
      system_instruction: systemInstruction
        ? { role: 'system', parts: [{ text: String(systemInstruction).slice(0, 20000) }] }
        : undefined,
      generationConfig: { ...baseCfg, candidateCount: 1, responseMimeType: 'text/plain' },
      safetySettings: [
        { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARM_CATEGORY_HATE_SPEECH',      threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARM_CATEGORY_HARASSMENT',        threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARM_CATEGORY_SEXUAL_CONTENT',    threshold: 'BLOCK_ONLY_HIGH' }
      ]
    });

    const once = async (model, timeoutMs = 35000) => {
      const url = `${BASE}/${model}:generateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`;
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), timeoutMs);
      try {
        const r = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json; charset=utf-8' },
          body: JSON.stringify(requestFor(model)),
          signal: ctrl.signal
        });
        const raw = await r.text();
        if (!r.ok) throw new Error(`${r.status} ${r.statusText} - ${raw.slice(0, 600)}`);
        const data = JSON.parse(raw);
        const text = data?.candidates?.[0]?.content?.parts?.map(p => p.text).join('') || '';
        if (!text) throw new Error('Empty response');
        return { text, modelUsed: model };
      } finally { clearTimeout(t); }
    };

    let lastErr = 'no-attempt';
    for (const model of PIPE) {
      for (let attempt = 1; attempt <= 2; attempt++) {
        try {
          const out = await once(model);
          return ok(out);
        } catch (e) {
          lastErr = `${model}#${attempt}: ${e.message || e}`;
          await new Promise(r => setTimeout(r, 400 * attempt));
        }
      }
    }
    return ok({ error: `All models failed: ${lastErr}` }, 502);
  } catch (e) {
    return ok({ error: String(e.message || e) }, 500);
  }
};
