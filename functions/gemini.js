// functions/gemini.js
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const MODEL = process.env.GEMINI_MODEL || 'gemini-1.5-flash';

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

  try {
    if (!GEMINI_API_KEY) return ok({ error: 'Missing GEMINI_API_KEY' }, 500);

    const { systemInstruction = '', history = [] } = JSON.parse(event.body || '{}');
    const contents = Array.isArray(history) ? history.map(m => ({
      role: m.role === 'user' ? 'user' : 'model',
      parts: (m.parts || []).map(p => ({ text: String(p.text || '').slice(0, 4000) }))
    })) : [];

    if (!contents.length || contents[contents.length - 1].role !== 'user') {
      contents.push({ role: 'user', parts: [{ text: 'ابدأ.' }] });
    }

    const body = {
      contents,
      system_instruction: systemInstruction ? { role: 'system', parts: [{ text: String(systemInstruction).slice(0, 12000) }] } : undefined,
      generationConfig: {
        temperature: 0.7, topK: 40, topP: 0.95, maxOutputTokens: 1024,
        responseMimeType: 'text/plain'
      },
      safetySettings: [
        { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HATE', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'HARASSMENT', threshold: 'BLOCK_ONLY_HIGH' },
        { category: 'SEXUAL', threshold: 'BLOCK_ONLY_HIGH' }
      ]
    };

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${encodeURIComponent(GEMINI_API_KEY)}`;
    const ctrl = typeof AbortController !== 'undefined' ? new AbortController() : null;
    if (ctrl?.signal) setTimeout(()=>ctrl.abort(), 20000);

    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
      body: JSON.stringify(body),
      signal: ctrl?.signal
    });

    if (!r.ok) {
      const errText = await r.text();
      return ok({ error: `Gemini error: ${r.status} ${r.statusText} - ${errText.slice(0, 600)}` }, 502);
    }

    const data = await r.json();
    const text = data?.candidates?.[0]?.content?.parts?.map(p => p.text).join('') || '';
    if (!text) return ok({ error: 'Empty response from model' }, 502);

    return ok({ text });
  } catch (e) {
    return ok({ error: String(e.message || e) }, 500);
  }
};
