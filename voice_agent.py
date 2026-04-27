"""
VitaAI Voice Agent — http://localhost:6000
Uses: Deepgram (STT) + Groq (AI) + VoiceRSS (TTS)
"""
import os, re, time, requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from urllib.parse import quote

load_dotenv()

DEEPGRAM_API_KEY  = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
OPENROUTER_KEY    = os.getenv("OPENROUTER_API_KEY")
VOICERSS_API_KEY = os.getenv("VOICERSS_API_KEY")

SYSTEM_PROMPT = """You are SyamHealth AI, a personal health coach and assistant.
Reply in 1-2 short sentences only. No bullet points, no symbols, no markdown.
Be friendly, warm, and give practical health advice when asked."""

chat_history = []
app = Flask(__name__)
CORS(app)


def transcribe(audio_bytes: bytes) -> str:
    r = requests.post(
        "https://api.deepgram.com/v1/listen?model=nova-2&language=en&punctuate=true&smart_format=true",
        headers={
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/webm;codecs=opus",
        },
        data=audio_bytes,
        timeout=30,
    )
    result = r.json()
    if "results" not in result:
        r2 = requests.post(
            "https://api.deepgram.com/v1/listen?model=nova-2&language=en&detect_language=true",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/ogg;codecs=opus",
            },
            data=audio_bytes,
            timeout=30,
        )
        result = r2.json()
        if "results" not in result:
            raise Exception(f"Deepgram error: {result.get('err_msg', str(result))}")

    transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
    return transcript.strip()


def ask_ai(user_text: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += chat_history[-10:]
    messages.append({"role": "user", "content": user_text})

    if GROQ_API_KEY:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 100, "temperature": 0.7},
                timeout=20
            )
            data = r.json()
            if "choices" in data:
                reply = data["choices"][0]["message"]["content"]
                reply = re.sub(r'[\*\#\_\`]', '', reply)
                return re.sub(r'\s+', ' ', reply).strip()
            print(f"Groq error: {data.get('error', {}).get('message', str(data))}")
        except Exception as e:
            print(f"Groq failed: {e}")

    if OPENROUTER_KEY:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json",
                     "HTTP-Referer": "http://localhost:3000", "X-Title": "VitaAI"},
            json={"model": "meta-llama/llama-3.1-8b-instruct:free", "messages": messages, "max_tokens": 100},
            timeout=20
        )
        data = r.json()
        if "choices" in data:
            reply = data["choices"][0]["message"]["content"]
            reply = re.sub(r'[\*\#\_\`]', '', reply)
            return re.sub(r'\s+', ' ', reply).strip()

    return "I'm having trouble connecting right now. Please try again in a moment."


def text_to_speech(text: str) -> bytes:
    """Convert text to WAV bytes using VoiceRSS TTS API."""
    r = requests.post(
        "https://api.voicerss.org/",
        data={
            "key": VOICERSS_API_KEY,
            "src": text,
            "hl":  "en-us",
            "v":   "Linda",       # voice: Linda, Amy, Mary, Mike, etc.
            "c":   "WAV",         # audio codec
            "f":   "16khz_16bit_mono",
            "r":   "0",           # speed: -10 (slow) to 10 (fast)
        },
        timeout=15,
    )

    # VoiceRSS returns an error as plain text starting with "ERROR"
    if r.content[:5] == b"ERROR" or r.headers.get("Content-Type", "").startswith("text"):
        raise Exception(f"VoiceRSS TTS error: {r.text.strip()}")

    return r.content   # raw WAV bytes


@app.route("/voice/chat", methods=["POST"])
def chat():
    audio_bytes = request.data
    print(f"📦 Received {len(audio_bytes)} bytes, Content-Type: {request.content_type}")

    if not audio_bytes or len(audio_bytes) < 500:
        return jsonify({"error": "Audio too short. Hold the button longer and speak clearly."}), 400

    try:
        user_text = transcribe(audio_bytes)
        print(f"👤 User: '{user_text}'")

        if not user_text:
            return jsonify({"error": "No speech detected. Speak clearly and try again."}), 400

        reply = ask_ai(user_text)
        print(f"🤖 Agent: {reply}")

        chat_history.append({"role": "user",      "content": user_text})
        chat_history.append({"role": "assistant", "content": reply})
        if len(chat_history) > 20:
            chat_history.pop(0); chat_history.pop(0)

        wav_data = text_to_speech(reply)

        response = app.response_class(wav_data, mimetype="audio/wav")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-User-Text"]   = quote(user_text)
        response.headers["X-Agent-Text"]  = quote(reply)
        response.headers["Access-Control-Expose-Headers"] = "X-User-Text, X-Agent-Text"
        return response

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":      "ok",
        "groq":        bool(GROQ_API_KEY),
        "openrouter":  bool(OPENROUTER_KEY),
        "deepgram":    bool(DEEPGRAM_API_KEY),
        "voicerss":    bool(VOICERSS_API_KEY),
    })


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 6000))

    print("=" * 52)
    print(f"  🎙️ VitaAI Voice Agent running on port {port}")
    print(f"  Groq:       {'✅' if GROQ_API_KEY else '❌ missing'}")
    print(f"  OpenRouter: {'✅' if OPENROUTER_KEY else '❌ missing'}")
    print(f"  Deepgram:   {'✅' if DEEPGRAM_API_KEY else '❌ missing'}")
    print(f"  VoiceRSS:   {'✅' if VOICERSS_API_KEY else '❌ missing'}")
    print("=" * 52)

    app.run(host="0.0.0.0", port=port, debug=False)
    