"""
╔══════════════════════════════════════════════════════════════════╗
║          🌐  GLOBAL AI CHATBOT  v3.0                            ║
╠══════════════════════════════════════════════════════════════════╣
║  STT   → Groq Whisper (whisper-large-v3)                        ║
║  LLM   → Groq LLaMA 3.3 70B  +  Smart Web Search Tool          ║
║  TTS   → Microsoft Edge TTS (edge-tts, 100% free)               ║
║  SEARCH→ DuckDuckGo (duckduckgo-search, cached + timeout)       ║
║  UI    → Terminal only (rich colored output)                     ║
╚══════════════════════════════════════════════════════════════════╝

v3.0 Changes (over v2.1):
  - Hybrid live-data classifier: fast regex pre-filter + lightweight
    LLM (llama-3.1-8b-instant) for ambiguous queries. Eliminates
    both false positives (unnecessary searches) and false negatives
    (missed current-event questions that regex couldn't catch).
  - Filler-word removal from transcriptions (from Code B).
  - Improved Hindi detection: Whisper tag + Devanagari scan + Roman
    Hindi keyword list (from Code B).
  - Context enrichment for short follow-up queries (from Code B).
  - is_only_fillers() guard prevents empty transcripts reaching LLM.
  - All Code A strengths preserved: startup self-test, GPIO, DDGS,
    search cache, timeout, Wikipedia fallback, error recovery, state
    machine, regex wake-word, history, Edge TTS.
"""

import os
import sys
import asyncio
import tempfile
import queue
import time
import threading
import re
import json
import logging
import socket
import concurrent.futures
from datetime import datetime
from zoneinfo import ZoneInfo
from enum import Enum
from typing import Optional, Tuple

# ── Wikipedia fallback ─────────────────────────────────────────────
try:
    import wikipedia as _wikipedia_module
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq
import edge_tts
import pygame
from dotenv import load_dotenv

# ── DuckDuckGo search ─────────────────────────────────────────────
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# ── GPIO (Raspberry Pi only) ──────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("globalbot")

# ─────────────────────────────────────────────────────────────────
#  TERMINAL COLORS
# ─────────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    RED     = "\033[91m"
    GREY    = "\033[90m"
    WHITE   = "\033[97m"

def banner(text: str, color: str = C.CYAN):
    width = 68
    print(f"\n{color}{C.BOLD}{'─' * width}")
    print(f"  {text}")
    print(f"{'─' * width}{C.RESET}")

def log_user(text: str, lang: str):
    tag = f"{C.BLUE}[YOU/{lang.upper()}]{C.RESET}"
    print(f"\n{tag}  {C.WHITE}{C.BOLD}{text}{C.RESET}")

def log_bot(text: str, lang: str):
    tag = f"{C.GREEN}[BOT/{lang.upper()}]{C.RESET}"
    print(f"{tag}  {C.CYAN}{text}{C.RESET}")

def log_search(query: str, mode: str = "web"):
    icon = "📰" if mode == "news" else "🔍"
    print(f"  {C.YELLOW}{icon} Searching [{mode}]:{C.RESET}  {query}")

def log_cache_hit(query: str):
    print(f"  {C.GREEN}⚡ Cache hit:{C.RESET}  {query[:80]}")

def log_result(snippet: str):
    preview = snippet[:120].replace("\n", " ")
    print(f"  {C.GREY}📄 Result:   {preview}...{C.RESET}")

def log_state(msg: str):
    print(f"  {C.MAGENTA}◆ {msg}{C.RESET}")

def log_warn(msg: str):
    print(f"  {C.RED}⚠  {msg}{C.RESET}")

def log_error(msg: str):
    print(f"  {C.RED}{C.BOLD}✖  {msg}{C.RESET}")

def log_pass(label: str):
    print(f"    {C.GREEN}✔  {label}{C.RESET}")

def log_fail(label: str):
    print(f"    {C.RED}✖  {label}{C.RESET}")

# ─────────────────────────────────────────────────────────────────
#  MODEL / API CONFIG
# ─────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    sys.exit("❌  GROQ_API_KEY not set. Add it to your .env file.")

CHAT_MODEL      = "llama-3.3-70b-versatile"
CLASSIFIER_MODEL = "llama-3.1-8b-instant"   # lightweight model for intent classification

STT_MODEL      = "whisper-large-v3"
STT_MODEL_FAST = "whisper-large-v3-turbo"

TTS_VOICE_EN   = "en-IN-NeerjaNeural"
TTS_VOICE_HI   = "hi-IN-SwaraNeural"

# ─────────────────────────────────────────────────────────────────
#  AUDIO CONFIG
# ─────────────────────────────────────────────────────────────────

SAMPLE_RATE          = 16_000
CHANNELS             = 1
CHUNK_SECS           = 0.1
ENERGY_THRESHOLD     = 0.010
SILENCE_AFTER_SPEECH = 0.8
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5

# ─────────────────────────────────────────────────────────────────
#  BEHAVIOUR CONFIG
# ─────────────────────────────────────────────────────────────────

MAX_TOKENS       = 400
CHAT_TEMPERATURE = 0.6

IDLE_TIMEOUT      = 10.0
IDLE_POLL_TIMEOUT = 60.0
HISTORY_WINDOW    = 10
SEARCH_TIMEOUT_SECS = 6

GREEN_LED_PIN = 18

# ─────────────────────────────────────────────────────────────────
#  WAKE WORDS
# ─────────────────────────────────────────────────────────────────

_WAKE_PATTERNS = [
    r"\bhello\b",
    r"\bhey\b",
    r"\bhi\b",
    r"\bokay\s+bot\b",
    r"\bhey\s+bot\b",
    r"\bglobalbot\b",
    r"\bbot\b",
]
_WAKE_REGEXES = [re.compile(p, re.IGNORECASE) for p in _WAKE_PATTERNS]

# ─────────────────────────────────────────────────────────────────
#  SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────

SYSTEM_EN = """\
You are GlobalBot, a friendly and knowledgeable AI assistant that can answer questions on ANY topic.
You have access to a web_search tool.

Use web_search ONLY when the question involves:
  - Current news, recent events, or breaking stories
  - Live data: weather, sports scores, stock prices, flight status
  - Facts that change over time (e.g. "Who is the current PM of India?")
  - Specific statistics or data you are not certain about

Do NOT use web_search for:
  - General knowledge, science, history, math, definitions
  - Conversational questions ("how are you", "tell me a joke")
  - Stable facts you already know with confidence

Rules for your spoken reply:
  - Keep every reply under 3 sentences. This is a voice interface.
  - Speak naturally — no bullet points, no markdown, no emoji.
  - Be concise but complete. Prioritise the single most useful fact.
  - Never refuse a general-knowledge question.
"""

SYSTEM_HI = """\
Aap GlobalBot hain — ek friendly AI assistant jo kisi bhi topic par jawab de sakta hai.
Aapke paas web_search tool hai.

web_search tab hi use karein jab sawaal ho:
  - Current news, recent events, breaking stories ke baare mein
  - Live data: mausam, sports scores, stock prices
  - Aisi jaankari jo time ke saath badlti hai
  - Specific facts jo aapko confirm karne hon

web_search mat karein:
  - General knowledge, science, history, math, definitions ke liye
  - Casual baatcheet ke liye
  - Stable facts ke liye jo aapko pehle se pata hain

Jawab dene ke niyam:
  - Roman/Latin script mein jawab dein — Devanagari bilkul mat use karein.
  - 3 sentence se zyada mat bolein. Yeh voice interface hai.
  - Koi bullet points, markdown ya emoji nahi.
"""

# ─────────────────────────────────────────────────────────────────
#  WEB SEARCH TOOL DEFINITION
# ─────────────────────────────────────────────────────────────────

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet for current, real-time, or factual information. "
            "Use ONLY for: current news, live sports scores, weather, stock prices, "
            "recent events, or facts that may have changed recently. "
            "Do NOT use for stable general knowledge questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A precise web search query string.",
                }
            },
            "required": ["query"],
        },
    },
}

# ─────────────────────────────────────────────────────────────────
#  NEWS QUERY DETECTION
# ─────────────────────────────────────────────────────────────────

_NEWS_KEYWORDS = re.compile(
    r"\b(news|latest|today|yesterday|breaking|headline|update|"
    r"current|recent|right now|this week|this month|score|"
    r"match|result|election|winner|died|arrested|launched|"
    r"announced|released|happened)\b",
    re.IGNORECASE,
)

def _is_news_query(query: str) -> bool:
    return bool(_NEWS_KEYWORDS.search(query))

# ─────────────────────────────────────────────────────────────────
#  HYBRID LIVE-DATA CLASSIFIER
#
#  Strategy: 3-tier decision pipeline
#
#  Tier A — DEFINITE NO (fast regex):
#    Pure greetings, math, definitions, basic science → skip search.
#    Cost: ~0 ms, 0 tokens.
#
#  Tier B — DEFINITE YES (fast regex):
#    Strong temporal/live-data signals → always search.
#    Cost: ~0 ms, 0 tokens.
#    Catches: "who won yesterday", "weather today", "latest NVIDIA GPU",
#    "what happened in Iran", "OpenAI release recently".
#
#  Tier C — AMBIGUOUS → LLM classifier (llama-3.1-8b-instant):
#    Queries that pass both regex tiers without a clear decision.
#    The small model answers YES/NO in ~0.3-0.5 s on Groq's fast lane.
#    Much cheaper than the full 70B model; adds minimal latency.
#    Cost: ~80-120 input tokens + 5 output tokens per ambiguous query.
#
#  Why not pure regex?
#    Regex misses paraphrased current-event questions:
#      "Tell me about Groq's newest models" (no temporal keyword)
#      "What is Iran situation" (no keyword, but clearly live)
#      "Latest NVIDIA cards" ("latest" is borderline in regex)
#
#  Why not pure LLM?
#    Adds ~0.5 s to every single query including "what is 2+2".
#    Wastes tokens; increases cost on repeated obvious queries.
#
#  The hybrid gives best-of-both: free instant decisions for clear
#  cases, smart LLM fallback only for genuinely ambiguous ones.
# ─────────────────────────────────────────────────────────────────

# Queries that clearly do NOT need live data
_DEFINITE_NO_RE = re.compile(
    r"^\s*("
    r"(what is|what are|define|explain|how does|how do|why is|why are|"
    r"tell me about|describe|who was|who were|when was|when did|"
    r"what was|history of|origin of|meaning of|difference between|"
    r"how to|can you|do you|are you|will you|could you)"
    r"\s+(?!.*\b(latest|current|now|today|recent|new|update|release|"
    r"happening|going on|right now|breaking|this year|2024|2025|2026)\b)"
    r")",
    re.IGNORECASE,
)

# Queries that clearly DO need live data — comprehensive
_DEFINITE_YES_RE = re.compile(
    r"\b("
    # Strong temporal signals
    r"today|tonight|tomorrow|yesterday|right now|just now|"
    r"this (morning|evening|afternoon|night|week|month|year)|"
    r"last (night|week|month|year|hour)|"
    r"currently|at the moment|as of now|"
    r"recent(ly)?|latest|newest|brand new|just (released|launched|announced|dropped)|"
    r"breaking|live|real.?time|"
    # Current office-holders — often missed by simple regex
    r"current (president|pm|prime minister|cm|chief minister|ceo|chairman|"
    r"captain|minister|governor|chancellor|director|head|leader)|"
    r"who (is|are) (the )?(current|now|today)|"
    r"who (leads|heads|runs|owns|controls) (the )?[a-z]|"
    # Tech/product releases — the biggest miss in Code A
    r"(latest|new|newest|recent|updated?) (model|version|release|update|"
    r"gpu|chip|phone|laptop|car|product|feature|api|tool)|"
    r"just (came out|released|launched|announced|dropped)|"
    r"what('?s| has| have) .{0,30} (released|launched|announced|done lately)|"
    # News / events
    r"news|headline|breaking|update|situation|happening|"
    r"what('?s| is) (going on|happening) in|"
    r"(war|conflict|crisis|protest|election|vote|result|score|match|"
    r"tournament|championship|final|semi.?final)|"
    r"(won|lost|beat|defeated|elected|appointed|resigned|arrested|"
    r"killed|died|passed away|launched|released|announced|signed)|"
    # Finance / weather
    r"(stock|share|price|market|sensex|nifty|nasdaq|dow|crypto|bitcoin|"
    r"weather|forecast|temperature|rain|flood|earthquake)|"
    # Hindi time/news words (Roman)
    r"aaj|kal|abhi|haal|taza khabar|abhi kya|kya ho raha"
    r")\b",
    re.IGNORECASE,
)

# Classifier system prompt — kept minimal for speed
_CLASSIFIER_SYSTEM = (
    "You are a query classifier. Respond with ONLY the word YES or NO.\n"
    "YES = the question requires current/live/recent internet information to answer correctly.\n"
    "NO  = the question can be answered from general knowledge.\n"
    "Examples:\n"
    "Q: What is photosynthesis? → NO\n"
    "Q: Who is the current CEO of OpenAI? → YES\n"
    "Q: Tell me about Groq's newest models. → YES\n"
    "Q: What is happening in Iran? → YES\n"
    "Q: How far is the moon from Earth? → NO\n"
    "Q: Latest NVIDIA GPU? → YES\n"
    "Q: Did OpenAI release anything recently? → YES\n"
    "Q: What is machine learning? → NO\n"
    "Q: Who won yesterday's IPL match? → YES\n"
    "Q: Tell me a joke. → NO\n"
)

def _llm_classify_live(query: str) -> bool:
    """
    Ask llama-3.1-8b-instant whether the query needs live internet data.
    Returns True (needs search) or False (can use LLM knowledge).
    Falls back to True on any error (safer to over-search than under-search).
    Target latency on Groq: ~300-500 ms.
    """
    try:
        resp = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user",   "content": f"Q: {query}"},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return answer.startswith("Y")
    except Exception as exc:
        logger.warning("LLM classifier failed, defaulting to YES: %s", exc)
        return True   # fail-safe: search rather than give stale answer


def requires_live_data(query: str, last_user_text: str = "") -> bool:
    """
    Hybrid live-data detector. Returns True if a web search should run.

    Pipeline:
      1. Definite-NO regex  → return False immediately.
      2. Definite-YES regex → return True immediately.
      3. Ambiguous          → ask llama-3.1-8b-instant (fast, cheap).

    The last_user_text context is appended to catch follow-ups like
    "did they win?" after a sports discussion.
    """
    # Build context-enriched text for classification
    full_text = query
    if last_user_text and len(query.split()) <= 6:
        # Short follow-up: enrich with prior context so classifier
        # can correctly classify "did they win?" as live-data
        full_text = f"{last_user_text} | {query}"

    # Tier A: clear NO
    if _DEFINITE_NO_RE.match(full_text):
        # Double-check: even stable-looking questions might carry live signals
        if not _DEFINITE_YES_RE.search(full_text):
            return False

    # Tier B: clear YES
    if _DEFINITE_YES_RE.search(full_text):
        return True

    # Tier C: ambiguous — use lightweight LLM classifier
    log_state("Ambiguous query — running LLM classifier...")
    return _llm_classify_live(full_text)


# ─────────────────────────────────────────────────────────────────
#  DATE/TIME QUERY DETECTION
# ─────────────────────────────────────────────────────────────────

_TIME_QUERY_RE = re.compile(
    r"\b("
    r"what time|current time|time is it|time now|"
    r"what('?s| is) today|today('?s)? date|today('?s)? day|"
    r"what day|which day|day of the week|"
    r"date today|current date|"
    r"what('?s| is) the date"
    r")\b",
    re.IGNORECASE,
)

def is_time_query(query: str) -> bool:
    return bool(_TIME_QUERY_RE.search(query))

def get_local_datetime() -> str:
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    return now.strftime("It is %-I:%M %p on %A, %-d %B %Y (IST).")

# ─────────────────────────────────────────────────────────────────
#  TRANSCRIPTION HELPERS  (from Code B — improved Hindi detection)
# ─────────────────────────────────────────────────────────────────

_FILLER_WORDS = {
    "hmm", "hmmm", "umm", "um", "uh", "uhh", "ah", "ahh",
    "mmm", "mm", "huh", "er", "erm", "accha", "acha", "achha",
    "haan", "han", "theek", "theek hai", "haanji",
    "mm-hmm", "mhm", "uh-huh", "okay", "ok", "right",
    "yeah", "yep", "yup",
}

_HINDI_ROMAN_WORDS = {
    "hai", "haan", "nahi", "kya", "kaise", "kitna", "kitni",
    "aap", "tum", "hum", "mein", "main", "kab", "ka", "ki",
    "kar", "karna", "sakta", "sakti", "jana", "chahiye",
    "batao", "bolo", "bata", "karo", "dono", "kyun", "kyunki",
    "lekin", "aur", "ya", "agar", "toh", "woh", "yeh", "iska",
    "uska", "inhe", "unhe", "accha", "theek", "bilkul",
}

def clean_transcription(text: str) -> str:
    """Remove standalone filler words from transcription."""
    words = text.split()
    cleaned = [w for w in words if w.lower().strip(".,!?") not in _FILLER_WORDS]
    return " ".join(cleaned).strip()

def is_only_fillers(text: str) -> bool:
    """Return True if the transcription contains nothing useful."""
    if not text:
        return True
    return clean_transcription(text).strip() == ""

# ─────────────────────────────────────────────────────────────────
#  SEARCH CACHE
# ─────────────────────────────────────────────────────────────────

search_cache: dict[str, str] = {}

def _cache_key(query: str) -> str:
    return query.lower().strip()

# ─────────────────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    SPEAKING  = "speaking"
    THINKING  = "thinking"

# ─────────────────────────────────────────────────────────────────
#  GPIO
# ─────────────────────────────────────────────────────────────────

def gpio_setup():
    if not GPIO_AVAILABLE:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
    GPIO.output(GREEN_LED_PIN, GPIO.LOW)

def gpio_set(on: bool):
    if GPIO_AVAILABLE:
        GPIO.output(GREEN_LED_PIN, GPIO.HIGH if on else GPIO.LOW)

def gpio_cleanup():
    if GPIO_AVAILABLE:
        GPIO.cleanup()

# ─────────────────────────────────────────────────────────────────
#  GROQ CLIENT
# ─────────────────────────────────────────────────────────────────

client   = Groq(api_key=GROQ_API_KEY)
_history = {"en": [], "hi": []}

# Per-language tracker for last user utterance — used for follow-up
# context enrichment (Code B idea) and classifier context.
_last_user_text: dict[str, str] = {"en": "", "hi": ""}

# ─────────────────────────────────────────────────────────────────
#  STARTUP SELF-TEST
# ─────────────────────────────────────────────────────────────────

def run_self_test() -> bool:
    banner("🔧  GlobalBot v3.0 — Startup Self-Test", C.YELLOW)
    all_ok = True

    # 1. Groq API
    try:
        test_resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        if test_resp.choices:
            log_pass(f"Groq API  ({CHAT_MODEL})")
        else:
            raise ValueError("Empty response")
    except Exception as exc:
        log_fail(f"Groq API — {exc}")
        all_ok = False

    # 2. Internet
    try:
        socket.setdefaulttimeout(4)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        log_pass("Internet connectivity")
    except Exception:
        log_fail("Internet connectivity — no network access")
        all_ok = False

    # 3. Microphone
    try:
        devices = sd.query_devices()
        input_devs = [d for d in devices if d["max_input_channels"] > 0]
        if input_devs:
            log_pass(f"Microphone ({input_devs[0]['name'][:40]})")
        else:
            raise RuntimeError("No input device found")
    except Exception as exc:
        log_fail(f"Microphone — {exc}")
        all_ok = False

    # 4. Speaker
    try:
        devices = sd.query_devices()
        output_devs = [d for d in devices if d["max_output_channels"] > 0]
        if output_devs:
            log_pass(f"Speaker   ({output_devs[0]['name'][:40]})")
        else:
            raise RuntimeError("No output device found")
    except Exception as exc:
        log_fail(f"Speaker — {exc}")
        all_ok = False

    # 5. DDGS
    if DDGS_AVAILABLE:
        log_pass("DuckDuckGo search library (DDGS)")
    else:
        log_warn("DDGS not installed — using Instant Answer API fallback")
        log_warn("  Run: pip install duckduckgo-search")

    # 6. Classifier model (quick smoke test)
    try:
        test_cls = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        if test_cls.choices:
            log_pass(f"Classifier model ({CLASSIFIER_MODEL})")
        else:
            raise ValueError("Empty response")
    except Exception as exc:
        log_fail(f"Classifier model — {exc}  (ambiguous queries will default to search)")

    print()
    if all_ok:
        print(f"  {C.GREEN}{C.BOLD}All systems ready.{C.RESET}\n")
    else:
        print(f"  {C.YELLOW}{C.BOLD}Some checks failed — bot will still try to run.{C.RESET}\n")

    return all_ok

# ─────────────────────────────────────────────────────────────────
#  VAD RECORDING
# ─────────────────────────────────────────────────────────────────

def capture_speech(timeout: float) -> Optional[np.ndarray]:
    audio_q   = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS,
        dtype="float32", blocksize=blocksize, callback=callback,
    )
    stream.start()
    gpio_set(True)

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
                    recording     = True
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
        gpio_set(False)

    if not speech_buffer:
        return None

    audio = np.concatenate(speech_buffer, axis=0)
    return audio if len(audio) >= SAMPLE_RATE * MIN_SPEECH_SECS else None

# ─────────────────────────────────────────────────────────────────
#  TRANSCRIBE  — improved Hindi detection + filler removal
# ─────────────────────────────────────────────────────────────────

def transcribe(audio: np.ndarray) -> Tuple[str, str]:
    return _transcribe_with_model(audio, STT_MODEL)

def transcribe_fast(audio: np.ndarray) -> Tuple[str, str]:
    return _transcribe_with_model(audio, STT_MODEL_FAST)

def _transcribe_with_model(audio: np.ndarray, model: str) -> Tuple[str, str]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio, SAMPLE_RATE)

    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="verbose_json",
            )
    finally:
        os.unlink(tmp_path)

    raw_text = (result.text or "").strip()

    # Guard: if transcription is only filler words, return empty
    if is_only_fillers(raw_text):
        return "", "en"

    # Clean filler words from the transcript
    text = clean_transcription(raw_text)

    # Language detection — Layer 1: Whisper reported language
    lang = (result.language or "en").strip().lower()
    if lang in ("ur", "ur-PK"):
        lang = "hi"
    if lang not in ("hi", "en"):
        lang = "en"

    # Layer 2: Devanagari / Arabic script scan
    for ch in raw_text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            lang = "hi"; break
        if 0x0600 <= cp <= 0x06FF:
            lang = "hi"; break

    # Layer 3: Common Roman Hindi keywords (from Code B)
    if any(w in _HINDI_ROMAN_WORDS for w in raw_text.lower().split()):
        lang = "hi"

    return text, lang

# ─────────────────────────────────────────────────────────────────
#  WAKE WORD
# ─────────────────────────────────────────────────────────────────

def is_wake_word(text: str) -> bool:
    for pattern in _WAKE_REGEXES:
        if pattern.search(text):
            return True
    return False

# ─────────────────────────────────────────────────────────────────
#  WEB SEARCH — cached, timed-out, news-routed, 4-tier fallback
# ─────────────────────────────────────────────────────────────────

def _ddgs_text_search(query: str) -> str:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            body = r.get("body", "")
            if body:
                results.append(f"{r.get('title', '')}: {body}")
    return " | ".join(results)[:1500] if results else ""


def _ddgs_news_search(query: str) -> str:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.news(query, max_results=5):
            title = r.get("title", "")
            body  = r.get("body", "") or r.get("excerpt", "")
            date  = r.get("date", "")
            if title:
                entry = f"[{date}] {title}: {body}" if date else f"{title}: {body}"
                results.append(entry)
    return " | ".join(results)[:1500] if results else ""


def _wikipedia_lookup(query: str) -> str:
    if not WIKIPEDIA_AVAILABLE:
        return ""
    try:
        titles = _wikipedia_module.search(query, results=3)
        if not titles:
            return ""
        summary = _wikipedia_module.summary(titles[0], sentences=3, auto_suggest=False)
        return f"[Wikipedia — {titles[0]}] {summary}"
    except _wikipedia_module.exceptions.DisambiguationError as e:
        try:
            summary = _wikipedia_module.summary(e.options[0], sentences=3, auto_suggest=False)
            return f"[Wikipedia — {e.options[0]}] {summary}"
        except Exception:
            return ""
    except Exception:
        logger.exception("Wikipedia lookup failed")
        return ""


_UNVERIFIED_WARNING_EN = (
    "Live information could not be verified through web search. "
    "The following answer is based on my general knowledge and may not reflect recent updates. "
)
_UNVERIFIED_WARNING_HI = (
    "Web search se live jaankari nahi mil payi. "
    "Yeh jawab meri general knowledge par based hai aur recent updates reflect nahi kar sakta. "
)


def do_web_search(query: str, lang: str = "en") -> Tuple[str, bool]:
    """
    4-tier search pipeline. Returns (result_text, search_succeeded).
    Tier 1: DDGS text/news | Tier 2: DDG Instant Answer API
    Tier 3: Wikipedia      | Tier 4: empty + False
    """
    import urllib.request, urllib.parse

    key = _cache_key(query)

    if key in search_cache:
        log_cache_hit(query)
        return search_cache[key], True

    result = ""

    # Tier 1: DDGS
    if DDGS_AVAILABLE:
        is_news   = _is_news_query(query)
        search_fn = _ddgs_news_search if is_news else _ddgs_text_search
        mode      = "news" if is_news else "web"
        log_search(query, mode)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(search_fn, query)
                result = future.result(timeout=SEARCH_TIMEOUT_SECS)
        except concurrent.futures.TimeoutError:
            log_warn(f"DDGS timed out after {SEARCH_TIMEOUT_SECS}s")
        except Exception as e:
            logger.exception("DDGS search failure")
            log_warn(f"DDGS search failed: {e}")

    # Tier 2: DDG Instant Answer API
    if not result:
        log_search(query, "instant")
        url = (
            "https://api.duckduckgo.com/?q="
            + urllib.parse.quote_plus(query)
            + "&format=json&no_html=1&skip_disambig=1"
        )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            abstract = data.get("AbstractText", "").strip()
            answer   = data.get("Answer", "").strip()
            topics   = data.get("RelatedTopics", [])
            if abstract:
                result = abstract[:800]
            elif answer:
                result = answer[:800]
            else:
                for t in topics:
                    if isinstance(t, dict) and t.get("Text"):
                        result = t["Text"][:800]
                        break
        except Exception as e:
            logger.exception("DDG Instant Answer API failure")
            log_warn(f"DDG Instant Answer failed: {e}")

    # Tier 3: Wikipedia
    if not result:
        log_search(query, "wikipedia")
        result = _wikipedia_lookup(query)
        if result:
            log_result(result)

    if not result:
        log_warn("All search tiers failed — LLM will answer with unverified warning")
        return "", False

    search_cache[key] = result
    log_result(result)
    return result, True

# ─────────────────────────────────────────────────────────────────
#  LLM REPLY  — with hybrid intent routing + follow-up enrichment
# ─────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"[*_`~^#\[\]{}<>•]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def get_ai_reply(user_text: str, lang: str) -> str:
    global _last_user_text

    # Route 1: Local clock (no network, no LLM)
    if is_time_query(user_text):
        log_state("Time query — answering from local clock (IST)")
        dt_str = get_local_datetime()
        if lang == "hi":
            return f"Abhi IST ke hisaab se: {dt_str}"
        return dt_str

    system       = SYSTEM_HI if lang == "hi" else SYSTEM_EN
    lang_history = _history[lang]

    # Context enrichment for short follow-up queries (from Code B)
    # e.g. "did they win?" after discussing a match
    contextual_input = user_text
    prior = _last_user_text.get(lang, "")
    if prior and len(user_text.split()) <= 6:
        contextual_input = f"[Follow-up to: {prior}] {user_text}"

    lang_history.append({"role": "user", "content": contextual_input})
    messages = [{"role": "system", "content": system}, *lang_history[-HISTORY_WINDOW:]]

    # Route 2: Hybrid live-data check
    if requires_live_data(user_text, last_user_text=prior):
        log_state("Live-data query — running web search pipeline")
        search_result, search_ok = do_web_search(user_text, lang)

        if search_ok:
            messages.append({
                "role":       "assistant",
                "content":    "",
                "tool_calls": [{
                    "id":       "forced_search_1",
                    "type":     "function",
                    "function": {
                        "name":      "web_search",
                        "arguments": json.dumps({"query": user_text}),
                    },
                }],
            })
            messages.append({
                "role":         "tool",
                "tool_call_id": "forced_search_1",
                "content":      search_result,
            })
        else:
            warning = _UNVERIFIED_WARNING_HI if lang == "hi" else _UNVERIFIED_WARNING_EN
            log_warn("Prepending unverified-data warning to LLM reply")
            messages.append({
                "role":    "user",
                "content": (
                    f"{warning}"
                    f"Please answer this question based on your general knowledge: {user_text}"
                ),
            })

    # Route 3: LLM call (also handles continuation of Route 2)
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=[WEB_SEARCH_TOOL],
            tool_choice="auto",
            max_tokens=MAX_TOKENS,
            temperature=CHAT_TEMPERATURE,
        )
        msg = response.choices[0].message

        # Tool-use loop (handles multi-call chains)
        while msg.tool_calls:
            messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                if tc.function.name == "web_search":
                    args  = json.loads(tc.function.arguments)
                    query = args.get("query", user_text)
                    result, _ = do_web_search(query, lang)
                    if not result:
                        result = "Web search unavailable right now."
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=CHAT_TEMPERATURE,
            )
            msg = response.choices[0].message

        reply = clean_text(msg.content or "")

    except Exception as exc:
        logger.exception("LLM call failed")
        log_error(f"LLM error: {exc}")
        reply = (
            "Sorry, I am having trouble right now. Please try again in a moment."
            if lang == "en" else
            "Maafi chahta hoon, abhi connection mein problem hai. Dobara try karein."
        )

    reply = reply or (
        "I could not find a good answer for that."
        if lang == "en" else
        "Is sawaal ka jawab abhi nahi mil paya."
    )

    lang_history.append({"role": "assistant", "content": reply})

    # Update last-user-text tracker for follow-up context
    _last_user_text[lang] = user_text

    return reply

# ─────────────────────────────────────────────────────────────────
#  TTS
# ─────────────────────────────────────────────────────────────────

pygame.mixer.init()

def pick_voice(lang: str, text: str = "") -> str:
    if lang == "hi":
        return TTS_VOICE_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI
    return TTS_VOICE_EN


async def _tts_save(text: str, path: str, voice: str):
    await edge_tts.Communicate(text, voice=voice).save(path)


def speak(text: str, lang: str = "en"):
    if not text.strip():
        return

    voice = pick_voice(lang, text)
    log_bot(text, lang)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    asyncio.run(_tts_save(text, tmp_path, voice))
    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)

    pygame.mixer.music.unload()
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

# ─────────────────────────────────────────────────────────────────
#  CHATBOT STATE MACHINE  — with strong error recovery
# ─────────────────────────────────────────────────────────────────

def chatbot_loop():
    state = State.LISTENING
    reply = ""
    lang  = "en"

    greeting = (
        "Hello! I'm GlobalBot, your AI assistant. "
        "I search the web when needed. Ask me anything!"
    )
    speak(greeting, "en")

    while True:
        try:

            # IDLE: wait for wake word
            if state == State.IDLE:
                log_state("IDLE — waiting for wake word...")
                audio = capture_speech(timeout=IDLE_POLL_TIMEOUT)
                if audio is None:
                    continue
                wake_text, _ = transcribe_fast(audio)
                log_user(wake_text, "en")
                if is_wake_word(wake_text):
                    state  = State.LISTENING
                    wakeup = "Haan, boliye." if lang == "hi" else "Yes, I'm listening."
                    speak(wakeup, lang)
                continue

            # LISTENING: capture and transcribe
            if state == State.LISTENING:
                log_state(f"Listening... (timeout {IDLE_TIMEOUT:.0f}s)")
                audio = capture_speech(timeout=IDLE_TIMEOUT)

                if audio is None:
                    state    = State.IDLE
                    idle_msg = (
                        "Idle mode mein ja raha hoon. Hello kahiye jab zaroorat ho."
                        if lang == "hi" else
                        "Going idle. Say Hello when you need me."
                    )
                    speak(idle_msg, lang)
                    continue

                log_state("Transcribing speech...")
                user_text, lang = transcribe(audio)

                if not user_text:
                    log_warn("Empty transcript — re-listening")
                    continue

                log_user(user_text, lang)
                log_state("Thinking...")
                reply = get_ai_reply(user_text, lang)
                state = State.SPEAKING
                continue

            # SPEAKING: speak reply, then return to listening
            if state == State.SPEAKING:
                speak(reply, lang)
                state = State.LISTENING
                continue

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            logger.exception("Unhandled exception in chatbot loop")
            log_error(f"Unexpected error in chatbot loop: {exc}")
            try:
                apology = (
                    "Something went wrong on my end. Please ask again."
                    if lang == "en" else
                    "Kuch gadbad ho gayi. Dobara poochein please."
                )
                speak(apology, lang)
            except Exception:
                logger.exception("speak() also failed during error recovery")
            state = State.LISTENING

# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main():
    gpio_setup()
    run_self_test()

    banner(
        f"🌐  GlobalBot v3.0  |  Model: {CHAT_MODEL}  |  "
        f"Search: {'DDGS' if DDGS_AVAILABLE else 'Fallback API'}  |  "
        f"Wiki: {'✔' if WIKIPEDIA_AVAILABLE else '✖'}  |  "
        f"Classifier: {CLASSIFIER_MODEL}",
        C.GREEN,
    )

    try:
        chatbot_loop()
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Interrupted by user.{C.RESET}")
    finally:
        gpio_cleanup()
        print(f"{C.GREEN}Shutdown complete.{C.RESET}\n")


if __name__ == "__main__":
    main()