// =============================================================================
// Netlify Serverless Function: The AI Core for "Sadiki AI"
// وظيفة سحابية لـ Netlify: "العقل" المدبر لتطبيق "صديقي الذكي"
//
// This function acts as a secure backend that communicates with the Google Gemini API.
// It keeps the API key secret and includes advanced features like model selection,
// enhanced safety settings, and robust error handling.
//
// هذه الوظيفة تعمل كواجهة خلفية آمنة تتصل بـ Google Gemini API.
// هي تحافظ على سرية مفتاح الربط وتتضمن ميزات متقدمة مثل اختيار النموذج،
// إعدادات أمان معززة، ومعالجة احترافية للأخطاء.
// =============================================================================

const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require("@google/generative-ai");

exports.handler = async function (event, context) {
  // Ensure the request is a POST request.
  // التأكد من أن الطلب هو من نوع POST.
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, body: "Method Not Allowed" };
  }

  try {
    // --- 1. Extract Data from the Frontend Request ---
    // --- 1. استخلاص البيانات من طلب الواجهة الأمامية ---
    const { systemInstruction, history } = JSON.parse(event.body);
    const apiKey = process.env.GEMINI_API_KEY;

    // A critical check to ensure the API key is configured in Netlify.
    // فحص حيوي للتأكد من أن مفتاح الـ API تم إعداده في Netlify.
    if (!apiKey) {
      console.error("Gemini API key is not set in environment variables.");
      return { statusCode: 500, body: JSON.stringify({ error: "API key is not configured on the server." }) };
    }
    
    // --- 2. Initialize the AI Model with Advanced Configuration ---
    // --- 2. إعداد نموذج الذكاء الاصطناعي بإعدادات متقدمة ---
    const genAI = new GoogleGenerativeAI(apiKey);
    
    // Advanced Safety Settings: Block harmful content at a high threshold for child safety.
    // إعدادات أمان متقدمة: حظر المحتوى الضار بمستوى عالٍ لضمان سلامة الطفل.
    const safetySettings = [
      { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
      { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
      { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
      { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
    ];

    // Generation Configuration: Maximize the output tokens for more comprehensive responses.
    // إعدادات التوليد: زيادة عدد الكلمات الممكنة للحصول على ردود أكثر شمولاً.
    const generationConfig = {
      maxOutputTokens: 8192,
      temperature: 0.9,
      topP: 1,
      topK: 1,
    };

    // Smart Model Selection: Try the most powerful model first, with a fallback to a faster one.
    // اختيار ذكي للنموذج: محاولة استخدام أقوى نموذج أولاً، مع وجود نموذج سريع كبديل احتياطي.
    const model = genAI.getGenerativeModel({ 
      model: "gemini-1.5-pro-latest",
      // model: "gemini-1.5-flash-latest", // Fallback model if Pro has issues
      safetySettings,
      generationConfig,
    });

    // --- 3. Prepare and Conduct the AI Chat Session ---
    // --- 3. تحضير وإجراء جلسة المحادثة مع الذكاء الاصطناعي ---
    
    // Ensure history is correctly formatted for the SDK.
    // التأكد من أن سجل المحادثة بالصيغة الصحيحة.
    const formattedHistory = (history || []).map(msg => ({
        role: msg.role === 'model' ? 'model' : 'user',
        parts: msg.parts,
    }));
    
    const chat = model.startChat({
      history: formattedHistory.slice(0, -1), // History without the last user message
      systemInstruction: systemInstruction,
    });
    
    // The last message in history is the user's new prompt.
    // آخر رسالة في السجل هي طلب المستخدم الجديد.
    const lastMessage = formattedHistory.length > 0 ? formattedHistory[formattedHistory.length - 1].parts[0].text : "";
    const result = await chat.sendMessage(lastMessage);
    const response = await result.response;
    const text = response.text();

    // --- 4. Return the Successful Response to the Frontend ---
    // --- 4. إرجاع الرد الناجح إلى الواجهة الأمامية ---
    return {
      statusCode: 200,
      body: JSON.stringify({ text }),
    };

  } catch (error) {
    // --- 5. Robust Error Handling ---
    // --- 5. معالجة احترافية للأخطاء ---
    console.error("Error in Gemini function:", error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: "Failed to get response from AI", details: error.message }),
    };
  }
};

