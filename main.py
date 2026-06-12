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
    • Multi-key fallback (up to 5 Groq API keys)
    • Internet / server connectivity announcements
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

# Groq exception classes (used for detecting rate-limit / quota errors).
# Wrapped in try/except so the script still runs even if the SDK's
# exception names ever change.
try:
    from groq import RateLimitError, APIConnectionError, APIStatusError
except ImportError:  # pragma: no cover - safety fallback
    RateLimitError = APIConnectionError = APIStatusError = Exception

load_dotenv()

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

# Up to 5 Groq API keys. Define GROQ_API_KEY_1 ... GROQ_API_KEY_5 in your
# .env file. For backward compatibility, GROQ_API_KEY (no suffix) is also
# accepted as one of the keys if the numbered ones aren't set.
_RAW_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
]

# Fall back to the old single GROQ_API_KEY if no numbered keys were given.
if not any(_RAW_KEYS) and os.getenv("GROQ_API_KEY"):
    _RAW_KEYS = [os.getenv("GROQ_API_KEY")]

GROQ_API_KEYS = [k for k in _RAW_KEYS if k]

if not GROQ_API_KEYS:
    raise RuntimeError(
        "No Groq API keys found. Please set GROQ_API_KEY_1 .. GROQ_API_KEY_5 "
        "(or GROQ_API_KEY) in your .env file."
    )

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

# Messages spoken for connectivity issues
NO_INTERNET_MESSAGE = {
    "en": "I can't connect to the internet. Please check your connection.",
    "hi": "Internet se connect nahi ho pa raha hai. Apna connection check karein.",
}

NO_SERVER_MESSAGE = {
    "en": "I can't connect to the server right now. Please try again later.",
    "hi": "Abhi server se connect nahi ho pa raha hai. Kripya thodi der baad try karein.",
}

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
#  CUSTOM EXCEPTIONS (connectivity)
# ──────────────────────────────────────────────

class NoInternetError(Exception):
    """Raised when there is no internet connection at all."""
    pass


class AllKeysExhaustedError(Exception):
    """Raised when every Groq API key has hit its rate limit / quota."""
    pass


# ──────────────────────────────────────────────
#  SETUP
# ──────────────────────────────────────────────

current_key_index = 0
client = Groq(api_key=GROQ_API_KEYS[current_key_index])

history = {"en": [], "hi": []}
last_user_text = ""

pygame.mixer.init()


# ══════════════════════════════════════════════
#  CONNECTIVITY HELPERS
# ══════════════════════════════════════════════

def has_internet(timeout: float = 3.0) -> bool:
    """Quick check for a working internet connection."""
    test_urls = [
        "https://www.google.com",
        "https://1.1.1.1",
        "https://api.groq.com",
    ]
    for url in test_urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GlobalBot/1.0"})
            urllib.request.urlopen(req, timeout=timeout)
            return True
        except Exception:
            continue
    return False


def _is_quota_or_rate_limit_error(e: Exception) -> bool:
    """Heuristic check for 'this key is out of quota / rate-limited' errors."""
    if isinstance(e, RateLimitError):
        return True

    msg = str(e).lower()
    keywords = [
        "rate limit", "rate_limit", "ratelimit",
        "quota", "429", "insufficient_quota",
        "exceeded", "too many requests",
    ]
    return any(k in msg for k in keywords)


def _is_connection_error(e: Exception) -> bool:
    """Heuristic check for network/connection failures (no internet)."""
    if isinstance(e, APIConnectionError):
        return True

    msg = str(e).lower()
    keywords = [
        "connection", "network", "timed out", "timeout",
        "name or service not known", "failed to resolve",
        "max retries exceeded", "temporary failure in name resolution",
    ]
    return any(k in msg for k in keywords)


def call_groq_with_fallback(call_fn):
    """
    Call `call_fn(client)` using the current Groq API key.
    If that key is rate-limited / out of quota, automatically rotate
    through the remaining keys.

    Raises:
        NoInternetError       - if there's no internet connection at all.
        AllKeysExhaustedError - if every API key is rate-limited/out of quota.
        Exception             - any other unexpected error is re-raised.
    """
    global current_key_index, client

    # First, make sure we actually have internet at all.
    if not has_internet():
        raise NoInternetError("No internet connection detected.")

    n = len(GROQ_API_KEYS)
    start = current_key_index
    last_quota_error: Optional[Exception] = None

    for offset in range(n):
        idx = (start + offset) % n
        try:
            active_client = Groq(api_key=GROQ_API_KEYS[idx])
            result = call_fn(active_client)

            # Success — remember this key as the current one.
            if idx != current_key_index:
                print(f"   🔄 Switched to Groq API key #{idx + 1}")
            current_key_index = idx
            client = active_client
            return result

        except Exception as e:
            if _is_quota_or_rate_limit_error(e):
                print(f"   ⚠️ API key #{idx + 1} exhausted/rate-limited: {e}")
                last_quota_error = e
                continue

            if _is_connection_error(e):
                # Could not reach the server even though basic internet
                # check passed — treat as a server connectivity issue.
                raise AllKeysExhaustedError(str(e)) from e

            # Any other error (auth issue, bad request, etc.) — re-raise.
            raise

    # All keys tried and all hit rate limit / quota errors.
    raise AllKeysExhaustedError(
        "All configured Groq API keys are rate-limited or out of quota."
    ) from last_quota_error


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
    """
    Transcribe audio and detect language (English or Hindi).

    Raises:
        NoInternetError       - if there's no internet connection.
        AllKeysExhaustedError - if every Groq API key is rate-limited/exhausted.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    sf.write(tmp_path, audio, SAMPLE_RATE)

    try:
        def _do_transcribe(active_client: Groq):
            with open(tmp_path, "rb") as f:
                return active_client.audio.transcriptions.create(
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

        result = call_groq_with_fallback(_do_transcribe)
    finally:
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
    """
    Get a reply from the LLM.

    Raises:
        NoInternetError       - if there's no internet connection.
        AllKeysExhaustedError - if every Groq API key is rate-limited/exhausted.
    """
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

    def _do_chat(active_client: Groq):
        return active_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                *lang_history,
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.4,
        )

    try:
        response = call_groq_with_fallback(_do_chat)
    except (NoInternetError, AllKeysExhaustedError):
        # Roll back the user message we just appended since we
        # weren't able to get a reply for it.
        lang_history.pop()
        raise

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
    print(f"  Loaded {len(GROQ_API_KEYS)} Groq API key(s) for fallback")
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
    reply = ""

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
                try:
                    wake_text, _ = transcribe(audio)
                except NoInternetError:
                    print("   ⚠️ No internet connection.")
                    speak(NO_INTERNET_MESSAGE["en"], lang="en")
                    continue
                except AllKeysExhaustedError:
                    print("   ⚠️ All API keys exhausted / server unreachable.")
                    speak(NO_SERVER_MESSAGE["en"], lang="en")
                    continue

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
                try:
                    user_text, lang = transcribe(audio)
                except NoInternetError:
                    print("   ⚠️ No internet connection.")
                    speak(NO_INTERNET_MESSAGE.get(lang, NO_INTERNET_MESSAGE["en"]), lang=lang)
                    continue
                except AllKeysExhaustedError:
                    print("   ⚠️ All API keys exhausted / server unreachable.")
                    speak(NO_SERVER_MESSAGE.get(lang, NO_SERVER_MESSAGE["en"]), lang=lang)
                    continue

                if not user_text:
                    print("⚠️  Could not understand — listening again.")
                    continue

                print(f"   You [{lang.upper()}] › {user_text}")

                print("🤔 Thinking...")
                try:
                    reply = get_ai_reply(user_text, lang)
                except NoInternetError:
                    print("   ⚠️ No internet connection.")
                    speak(NO_INTERNET_MESSAGE.get(lang, NO_INTERNET_MESSAGE["en"]), lang=lang)
                    continue
                except AllKeysExhaustedError:
                    print("   ⚠️ All API keys exhausted / server unreachable.")
                    speak(NO_SERVER_MESSAGE.get(lang, NO_SERVER_MESSAGE["en"]), lang=lang)
                    continue

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
