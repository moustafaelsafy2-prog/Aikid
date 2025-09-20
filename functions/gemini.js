// functions/gemini.js
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const CUSTOM_MODEL   = process.env.GEMINI_MODEL; // اختياري
const CANDIDATES = [
  'gemini-1.5-pro',
  'gemini-1.5-flash'
];

if (CUSTOM_MODEL && !CANDIDATES.includes(CUSTOM_MODEL)) CANDIDATES.unshift(CUSTOM_MODEL);

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
    const { systemInstruction = '', history = [] } = JSON.parse(event.body || '{}');

    // تهيئة التاريخ للموديل
    const contents = Array.isArray(history) ? history.map(m => ({
      role: m.role === 'user' ? 'user' : 'model',
      parts: (m.parts || []).map(p => ({ text: String(p.text || '').slice(0, 8000) }))
    })) : [];

    if (!contents.length || contents[contents.length - 1].role !== 'user') {
      contents.push({ role: 'user', parts: [{ text: 'ابدأ.' }] });
    }

    const payload = (model) => ({
      contents,
      system_instruction: systemInstruction ? { role: 'system', parts: [{ text: String(systemInstruction).slice(0, 20000) }] } : undefined,
      generationConfig: {
        temperature: 0.6,      // دقة أعلى واستقرار
        topK: 40,
        topP: 0.9,
        maxOutputTokens: 1400,
        candidateCount: 1,
        responseMimeType: 'text/plain'
      },
      safetySettings: [
        { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARM_CATEGORY_SEXUAL_CONTENT', threshold: 'BLOCK_ONLY_HIGH' }
      ],
      _model: model
    });

    const tryOnce = async (model, attempt, timeoutMs=20000) => {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`;
      const ctrl = new AbortController();
      const timer = setTimeout(()=>ctrl.abort(), timeoutMs);
      try {
        const r = await fetch(url, { method: 'POST', headers: {'Content-Type':'application/json; charset=utf-8'}, body: JSON.stringify(payload(model)), signal: ctrl.signal });
        const txt = await r.text();
        if (!r.ok) throw new Error(`${r.status} ${r.statusText} - ${txt.slice(0, 600)}`);
        const data = JSON.parse(txt);
        const out = data?.candidates?.[0]?.content?.parts?.map(p=>p.text).join('') || '';
        if (!out) throw new Error('Empty response');
        return out;
      } finally { clearTimeout(timer); }
    };

    // محاولات + backoff
    let lastErr = null;
    for (let i = 0; i < CANDIDATES.length; i++) {
      const model = CANDIDATES[i];
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const text = await tryOnce(model, attempt);
          return ok({ text });
        } catch (e) {
          lastErr = `Model ${model} attempt ${attempt+1}: ${e.message||e}`;
          await new Promise(r=>setTimeout(r, 400 * (attempt+1))); // backoff بسيط
        }
      }
    }
    return ok({ error: `All models failed. Last: ${lastErr}` }, 502);
  } catch (e) {
    return ok({ error: String(e.message || e) }, 500);
  }
};
