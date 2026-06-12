"""
============================================================
  🌐 GlobalBot — Simple Speech-to-Speech AI Assistant
============================================================
  Stack:
    STT    → Groq Whisper (whisper-large-v3)
    LLM    → Groq LLaMA 3.3 70B
    TTS    → Microsoft Edge TTS (edge-tts, free)
    SEARCH → DuckDuckGo API (urllib, zero extra packages)

  Features:
    • Answers all types of questions (science, history, tech, etc.)
    • Live web search for current events and news
    • Auto language detection: English ↔ Hindi
    • Idle mode after 10 seconds of silence
============================================================
"""

import os
import asyncio
import tempfile
import queue
import time
import re
import urllib.request
import urllib.parse
import json
import html
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq
import edge_tts
import pygame
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

STT_MODEL  = "whisper-large-v3"
CHAT_MODEL = "llama-3.3-70b-versatile"

TTS_VOICE_EN = "en-US-JennyNeural"
TTS_VOICE_HI = "hi-IN-SwaraNeural"

SAMPLE_RATE = 16000
CHANNELS    = 1
MAX_TOKENS  = 400

ENERGY_THRESHOLD     = 0.035
SILENCE_AFTER_SPEECH = 0.8
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.25

IDLE_TIMEOUT      = 10.0   # seconds of silence before going idle
IDLE_POLL_TIMEOUT = 30.0   # how long to wait in idle before polling again

WAKE_WORDS = ["hello", "hey", "globalbot", "hello bot", "hey bot",
              "hello globalbot", "hey globalbot"]

# Keywords that suggest the question needs a live web search
SEARCH_TRIGGERS = [
    # English
    "latest", "recent", "current", "today", "yesterday", "this week",
    "this month", "this year", "2024", "2025", "2026", "news", "update",
    "score", "result", "match", "election", "winner", "died", "launched",
    "released", "announced", "price", "stock", "weather", "live",
    "who is", "what is the latest", "what happened",
    # Hindi (Roman)
    "aaj", "kal", "abhi", "haal", "taza", "khabar", "result",
    "score", "match", "chunav", "mausam",
]

# ──────────────────────────────────────────────
#  SYSTEM PROMPTS
# ──────────────────────────────────────────────

SYSTEM_EN = """
M A B F M  school is one of the top schools of khajrana indore which is located near dargah main road. 
YOU ARE MADE BY ROBOTWALA .
You are a simple AI assistant.

Rules:
- Answer only the user's question.
- Keep answers very short: maximum 1-2 sentences.
- Do not give long explanations unless explicitly asked.
- Be direct and accurate.
- No bullet points or markdown formatting.

"""

SYSTEM_HI = """
M A B F M School khajrana ka ek mash-hoor school hai , jo ki dargah main road pr sthit hai .
Aap ek simple AI assistant hain.
TUMHE ROBOTWALA NE BANAYA HAI . 
Rules:
- Sirf user ke question ka jawab dein.
- Jawab bahut chhota rakhein: maximum 1-2 sentences.
- Jab tak user na kahe, lambi explanation mat dein.
- Seedha aur accurate jawab dein.
- Koi bullet points ya markdown formatting mat use karein.
"""

# ──────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────

class State(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    SPEAKING  = "speaking"
    

# ──────────────────────────────────────────────
#  SETUP
# ──────────────────────────────────────────────

client = Groq(api_key=GROQ_API_KEY)

history = {"en": [], "hi": []}
last_user_text = ""

pygame.mixer.init()


# ══════════════════════════════════════════════
#  WEB SEARCH
# ══════════════════════════════════════════════

def needs_web_search(text: str) -> bool:
    lower = text.lower()
    return any(trigger in lower for trigger in SEARCH_TRIGGERS)


def web_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo Instant Answer API using built-in urllib."""
    try:
        print(f"   🔍 Searching: {query!r}")
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        })
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "GlobalBot/1.0"})

        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []

        abstract = data.get("AbstractText", "").strip()
        if abstract:
            source = data.get("AbstractSource", "")
            results.append(f"{source}: {abstract}" if source else abstract)

        answer = data.get("Answer", "").strip()
        if answer:
            results.append(f"Direct answer: {answer}")

        for topic in data.get("RelatedTopics", [])[:max_results]:
            text = topic.get("Text", "").strip()
            if text and len(text) > 20:
                results.append(html.unescape(text))
            for sub in topic.get("Topics", [])[:2]:
                sub_text = sub.get("Text", "").strip()
                if sub_text and len(sub_text) > 20:
                    results.append(html.unescape(sub_text))

        if not results:
            results = _ddg_lite_search(query, max_results)

        if not results:
            return ""

        print(f"   ✅ Got {len(results)} result(s)")
        return "\n\n".join(results[:max_results])

    except Exception as e:
        print(f"   ⚠️ Search failed: {e}")
        return ""


def _ddg_lite_search(query: str, max_results: int = 4) -> list:
    """Fallback: scrape DuckDuckGo lite (plain HTML, no JS)."""
    try:
        params = urllib.parse.urlencode({"q": query, "kl": "in-en"})
        url = f"https://lite.duckduckgo.com/lite/?{params}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })
        with urllib.request.urlopen(req, timeout=6) as resp:
            page = resp.read().decode("utf-8", errors="ignore")

        snippets = []
        marker = "result-snippet"
        pos = 0
        while len(snippets) < max_results:
            idx = page.find(marker, pos)
            if idx == -1:
                break
            start = page.find(">", idx) + 1
            end   = page.find("</td>", start)
            if start == 0 or end == -1:
                pos = idx + 1
                continue
            raw = page[start:end].strip()
            clean = re.sub(r"<[^>]+>", "", raw).strip()
            clean = html.unescape(clean)
            if len(clean) > 20:
                snippets.append(clean)
            pos = end

        return snippets
    except Exception:
        return []


def build_system_with_search(query: str, lang: str) -> str:
    """Inject web search results into the system prompt if needed."""
    base = SYSTEM_HI if lang == "hi" else SYSTEM_EN

    if not needs_web_search(query):
        return base

    context = web_search(query)
    if not context:
        return base

    if lang == "hi":
        inject = (
            "\n\nWeb search se mili latest jaankari (isko apne jawab mein use karo):\n\n"
            + context
        )
    else:
        inject = (
            "\n\nWeb search results for this query (use this to answer accurately):\n\n"
            + context
        )

    return base + inject


# ══════════════════════════════════════════════
#  AUDIO CAPTURE
# ══════════════════════════════════════════════

def capture_speech(timeout: float) -> Optional[np.ndarray]:
    """Record audio until silence is detected or timeout is reached."""
    audio_q   = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )
    stream.start()

    speech_buffer: list            = []
    pre_buffer:    list            = []
    recording                      = False
    silence_start: Optional[float] = None
    idle_clock                     = time.time()

    try:
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                if not recording and time.time() - idle_clock >= timeout:
                    return None
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= ENERGY_THRESHOLD:
                idle_clock    = time.time()
                silence_start = None

                if not recording:
                    recording = True
                    speech_buffer = list(pre_buffer)

                speech_buffer.append(chunk)

            elif recording:
                speech_buffer.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_AFTER_SPEECH:
                    break

            else:
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)

                if time.time() - idle_clock >= timeout:
                    return None

    finally:
        stream.stop()
        stream.close()

    if not speech_buffer:
        return None

    audio = np.concatenate(speech_buffer, axis=0)
    return audio if len(audio) >= SAMPLE_RATE * MIN_SPEECH_SECS else None


# ══════════════════════════════════════════════
#  TRANSCRIPTION
# ══════════════════════════════════════════════

def transcribe(audio: np.ndarray) -> Tuple[str, str]:
    """Transcribe audio and detect language (English or Hindi)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    sf.write(tmp_path, audio, SAMPLE_RATE)

    with open(tmp_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            temperature=0,
            prompt=(
                "This is a general-purpose AI assistant. "
                "The user may ask about any topic in English or Hindi. "
                "Use Roman Hindi for Hindi speech. "
                "Transcribe accurately — do not skip words."
            ),
            response_format="verbose_json",
        )

    os.unlink(tmp_path)

    raw_text = (result.text or "").strip()
    text = clean_transcription(raw_text)

    if is_only_fillers(raw_text):
        return "", "en"

    # Language detection — Layer 1: Whisper tag
    lang = (result.language or "en").strip().lower()
    if lang == "ur":
        lang = "hi"
    if lang not in ("hi", "en"):
        lang = "en"

    # Layer 2: Devanagari / Arabic script scan
    for ch in raw_text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            lang = "hi"
            break

    # Layer 3: Common Hindi words in Roman script
    hindi_words = {
        "hai", "haan", "nahi", "kya", "kaise", "kitna", "kitni",
        "aap", "tum", "hum", "mein", "main", "kab", "ka", "ki",
        "kar", "karna", "sakta", "sakti", "jana", "chahiye",
        "batao", "bolo", "bata", "karo", "dono", "kyun", "kyunki",
        "lekin", "aur", "ya", "agar", "toh", "woh", "yeh", "iska",
        "uska", "inhe", "unhe", "accha", "theek", "bilkul",
    }
    if any(w in hindi_words for w in raw_text.lower().split()):
        lang = "hi"

    return text, lang


def clean_transcription(text: str) -> str:
    """Remove filler words from transcription."""
    fillers = {
        "hmm", "hmmm", "umm", "um", "uh", "uhh", "ah", "ahh",
        "mmm", "mm", "huh", "er", "erm", "accha", "acha", "achha",
        "haan", "han", "theek", "theek hai", "haanji",
        "mm-hmm", "mhm", "uh-huh", "okay", "ok", "right",
        "yeah", "yep", "yup",
    }
    words = text.split()
    cleaned = [w for w in words if w.lower().strip(".,!?") not in fillers]
    return " ".join(cleaned).strip()


def is_only_fillers(text: str) -> bool:
    if not text:
        return True
    return clean_transcription(text).strip() == ""


# ══════════════════════════════════════════════
#  WAKE WORD
# ══════════════════════════════════════════════

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)


# ══════════════════════════════════════════════
#  AI REPLY
# ══════════════════════════════════════════════

def get_ai_reply(user_text: str, lang: str) -> str:
    global last_user_text

    lang_history = history[lang]

    # Enrich short follow-up queries with prior context
    contextual_input = user_text
    if last_user_text and len(user_text.split()) <= 6:
        contextual_input = (
            f"Previous question: {last_user_text}\n"
            f"Follow-up: {user_text}"
        )

    system = build_system_with_search(contextual_input, lang)

    lang_history.append({"role": "user", "content": contextual_input})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            *lang_history,
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.4,
    )

    reply = response.choices[0].message.content.strip()

    # Strip markdown formatting (shouldn't appear in TTS output)
    reply = re.sub(r'\*+', '', reply)
    reply = re.sub(r'#+\s*', '', reply)
    reply = re.sub(r'`+', '', reply)

    lang_history.append({"role": "assistant", "content": reply})
    last_user_text = user_text

    # Keep history bounded to last 10 exchanges
    if len(lang_history) > 20:
        history[lang] = lang_history[-20:]

    return reply


# ══════════════════════════════════════════════
#  TTS / SPEAK
# ══════════════════════════════════════════════

def pick_voice(text: str, lang: str) -> str:
    if lang == "hi":
        return TTS_VOICE_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI
    return TTS_VOICE_EN


async def _tts(text: str, path: str, voice: str):
    text = str(text).strip()
    if not text:
        text = "Sorry, I could not understand."
    text = text.replace("*", "").replace("#", "").replace("`", "")
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(path)
    except Exception as e:
        print(f"⚠️ TTS Error: {e}")
        fallback = edge_tts.Communicate(
            text="Sorry, there was a voice generation problem.",
            voice="en-US-JennyNeural",
        )
        await fallback.save(path)


def speak(text: str, lang: str = "en"):
    """Convert text to speech and play it. Blocks until playback completes."""
    voice = pick_voice(text, lang)
    print(f"   🔊 Voice → {voice}")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        asyncio.run(_tts(text, tmp_path, voice))
    except Exception as e:
        print(f"⚠️ Async TTS Error: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return

    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()

    # Wait for playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.wait(50)

    pygame.mixer.music.unload()
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════

def print_banner():
    print("\n" + "=" * 60)
    print("  🌐 GlobalBot — Your All-Knowing AI Assistant")
    print("=" * 60)
    print("  Live web search for current events")
    print("  States:")
    print("    👂 LISTENING  — auto-detects your voice")
    print(f"    😴 IDLE       — {int(IDLE_TIMEOUT)}s silence → idle")
    print("                   say 'Hello' to wake up")
    print("  Ctrl+C to quit")
    print("=" * 60 + "\n")


def state_label(state: State) -> str:
    return {
        State.IDLE:      "😴 IDLE",
        State.LISTENING: "👂 LISTENING",
        State.SPEAKING:  "🔊 SPEAKING",
    }[state]


# ══════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════

def main():
    print_banner()

    state = State.LISTENING
    lang  = "en"

    speak(
        "Hello! I'm GlobalBot, your all-knowing AI assistant. "
        "What would you like to know?",
        lang="en",
    )

    try:
        while True:

            # ── IDLE ──────────────────────────────────
            if state == State.IDLE:
                print(f"\n{state_label(state)}  — say 'Hello' to activate...")

                audio = capture_speech(timeout=IDLE_POLL_TIMEOUT)
                if audio is None:
                    continue

                print("🔍 Checking for wake word...")
                wake_text, _ = transcribe(audio)
                print(f"   Heard: {wake_text!r}")

                if is_wake_word(wake_text):
                    state = State.LISTENING
                    print("\n✅ Wake word detected!")
                    speak("Hello! I'm listening. What would you like to know?", lang="en")
                else:
                    print("   Not a wake word — staying idle.")
                continue

            # ── LISTENING ─────────────────────────────
            if state == State.LISTENING:
                print(f"\n{state_label(state)}  — silence for {int(IDLE_TIMEOUT)}s → idle")

                audio = capture_speech(timeout=IDLE_TIMEOUT)

                if audio is None:
                    state = State.IDLE
                    print(f"\n⏱️  No speech for {int(IDLE_TIMEOUT)}s — going idle.")
                    speak("Going to sleep now. Say 'Hello' when you need me.", lang="en")
                    continue

                print("🔍 Transcribing...")
                user_text, lang = transcribe(audio)

                if not user_text:
                    print("⚠️  Could not understand — listening again.")
                    continue

                print(f"   You [{lang.upper()}] › {user_text}")

                print("🤔 Thinking...")
                reply = get_ai_reply(user_text, lang)
                print(f"   AI  [{lang.upper()}] › {reply}")

                state = State.SPEAKING
                continue

            # ── SPEAKING ──────────────────────────────
            if state == State.SPEAKING:
                print(f"\n{state_label(state)}")
                speak(reply, lang)
                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down GlobalBot. Goodbye!")


if __name__ == "__main__":
    main()
