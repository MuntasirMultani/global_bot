"""
============================================================
  🤖 Speech-to-Speech AI Chatbot — Powered by Groq (Free)
============================================================
  Stack:
    STT  → Groq Whisper (whisper-large-v3)
    LLM  → Groq LLaMA 3.3 70B
    TTS  → Microsoft Edge TTS (edge-tts, 100% free)

  Language Support:
    → Speak English → AcroBot replies & speaks in English
    → Speak Hindi   → AcroBot replies & speaks in Hindi
    → Switches instantly every message — no confusion

  Language Detection (3-layer):
    1. Whisper language tag  (fast, sometimes wrong)
    2. Script scan of transcript  (ground truth — never lies)
    3. Default → English

  State Machine:
    IDLE  ──(wake word)──►  LISTENING  ──(speech)──►  SPEAKING
      ▲                          │                        │
      └────────(10s silence)─────┘◄───────────────────────┘

  Wake word: "Hello" (or just "Acrobot")
============================================================
"""

import os
import asyncio
import tempfile
import queue
import time
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

TTS_VOICE_EN = "en-US-JennyNeural"   # change to en-US-GuyNeural for male
TTS_VOICE_HI = "hi-IN-SwaraNeural"  # change to hi-IN-MadhurNeural for male

SAMPLE_RATE = 16000
CHANNELS    = 1
MAX_TOKENS  = 300

# ── VAD tuning ─────────────────────────────────────────────
ENERGY_THRESHOLD     = 0.035   # RMS level above which audio counts as "speech"
                               # ↑ raise (e.g. 0.025) if background noise triggers false starts
                               # ↓ lower (e.g. 0.010) if mic is quiet and misses your voice

SILENCE_AFTER_SPEECH = 1.5   # seconds of silence that marks end-of-tur
PRE_ROLL_CHUNKS      = 6      # chunks buffered before speech onset (avoids clipping first word)
MIN_SPEECH_SECS      = 0.5    # discard clips shorter than this (accidental noise / breath)
CHUNK_SECS           = 0.1    # size of each audio chunk in seconds

IDLE_TIMEOUT         = 10.0   # secs of no speech in LISTENING → go IDLE
IDLE_POLL_TIMEOUT    = 30.0   # how long to wait for audio in IDLE before re-looping

# ── Wake words (lowercase, substring match) ────────────────
WAKE_WORDS = ["hello", "hey"]

SYSTEM_EN = (
    "Your name is RoboBot. You are the helpful AI assistant . "
    "Keep responses concise and conversational."
    "No bullet points or markdown."
)

SYSTEM_HI = (
    "Aapka naam RoboBot hai. Aap helpful AI assistant hain. "
    "Apne uttar chhote aur batcheet ke andaz mein rakhein. "
    "Koi bullet points ya markdown nahi."
)

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

# Separate history per language so the model never sees
# cross-language context and stays in the right language naturally.
history: dict = {
    "en": [],
    "hi": [],
}

pygame.mixer.init()


# ──────────────────────────────────────────────
#  VAD RECORDING
# ──────────────────────────────────────────────

def capture_speech(timeout: float) -> Optional[np.ndarray]:
    """
    Listens via microphone using Voice Activity Detection.

    Returns audio ndarray when:
      • Speech is detected AND then silence >= SILENCE_AFTER_SPEECH seconds.

    Returns None when:
      • No speech detected for `timeout` seconds (caller decides what to do).

    How it works:
      1. Continuously reads 100ms audio chunks into a queue.
      2. Computes RMS energy per chunk.
      3. Above ENERGY_THRESHOLD  → "speech": start/continue recording.
      4. Below threshold after speech began → silence timer starts.
      5. Silence timer expires              → user finished speaking, return audio.
      6. No speech at all for `timeout`     → return None.
    """
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
    pre_buffer:    list            = []   # rolling window before speech onset
    recording                      = False
    silence_start: Optional[float] = None
    idle_clock                     = time.time()

    try:
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                # Check idle timeout even when mic is completely silent
                if not recording and time.time() - idle_clock >= timeout:
                    return None
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= ENERGY_THRESHOLD:
                # ── Speech detected ──────────────────────────────────
                idle_clock    = time.time()   # reset the no-speech clock
                silence_start = None

                if not recording:
                    recording = True
                    # Prepend pre-roll so the first syllable isn't clipped
                    speech_buffer = list(pre_buffer)

                speech_buffer.append(chunk)

            elif recording:
                # ── Silence after speech has started ─────────────────
                speech_buffer.append(chunk)   # keep trailing silence for natural cut
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_AFTER_SPEECH:
                    break                     # end of user's turn → exit loop

            else:
                # ── Still waiting for speech ─────────────────────────
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)

                if time.time() - idle_clock >= timeout:
                    return None               # timed out with no speech

    finally:
        stream.stop()
        stream.close()

    if not speech_buffer:
        return None

    audio = np.concatenate(speech_buffer, axis=0)
    return audio if len(audio) >= SAMPLE_RATE * MIN_SPEECH_SECS else None


# ──────────────────────────────────────────────
#  TRANSCRIBE
# ──────────────────────────────────────────────

def transcribe(audio: np.ndarray) -> Tuple[str, str]:
    """
    Returns (text, lang_code) where lang_code is 'hi' or 'en'.

    Language detection uses 3 layers in order:
      1. Whisper's detected language tag  (quick but sometimes wrong)
      2. Script scan of the transcribed text  ← THE KEY FIX
         Whisper always transcribes the correct script even when
         its language tag is wrong. Devanagari/Arabic in the text
         means Hindi, period.
      3. Default → 'en'
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    sf.write(tmp_path, audio, SAMPLE_RATE)

    with open(tmp_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            response_format="verbose_json",
        )

    os.unlink(tmp_path)

    text = (result.text or "").strip()
    lang = (result.language or "en").strip().lower()

    # Layer 1: normalise Whisper tag
    if lang == "ur":
        lang = "hi"
    if lang not in ("hi", "en"):
        lang = "en"

    # Layer 2: script scan — overrides the tag if script is Hindi
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:   # Devanagari script
            lang = "hi"
            break
        if 0x0600 <= cp <= 0x06FF:   # Arabic / Urdu script
            lang = "hi"
            break

    return text, lang


# ──────────────────────────────────────────────
#  WAKE WORD
# ──────────────────────────────────────────────

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)


# ──────────────────────────────────────────────
#  AI REPLY
# ──────────────────────────────────────────────

def get_ai_reply(user_text: str, lang: str) -> str:
    system       = SYSTEM_HI if lang == "hi" else SYSTEM_EN
    lang_history = history[lang]

    lang_history.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            *lang_history,
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.7,
    )

    reply = response.choices[0].message.content.strip()
    lang_history.append({"role": "assistant", "content": reply})
    return reply


# ──────────────────────────────────────────────
#  VOICE SELECTION
# ──────────────────────────────────────────────

def pick_voice(text: str, lang: str) -> str:
    if lang == "hi":
        return TTS_VOICE_HI

    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return TTS_VOICE_HI
        if 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI

    return TTS_VOICE_EN


# ──────────────────────────────────────────────
#  SPEAK
# ──────────────────────────────────────────────

async def _tts(text: str, path: str, voice: str):
    await edge_tts.Communicate(text, voice=voice).save(path)


def speak(text: str, lang: str = "en"):
    voice = pick_voice(text, lang)
    print(f"   🔊 Voice → {voice}")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    asyncio.run(_tts(text, tmp_path, voice))

    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    pygame.mixer.music.unload()
    os.unlink(tmp_path)


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def print_banner():
    print("\n" + "=" * 56)
    print("=" * 56)
    print("  States:")
    print("    👂 LISTENING  — auto-detects your voice")
    print(f"    😴 IDLE       — {int(IDLE_TIMEOUT)}s silence → idle")
    print("                   say 'Hello' to wake up")
    print("    🔊 SPEAKING   — playing response; loops back after")
    print("  Ctrl+C to quit")
    print("=" * 56 + "\n")


def state_label(state: State) -> str:
    return {
        State.IDLE:      "😴 IDLE",
        State.LISTENING: "👂 LISTENING",
        State.SPEAKING:  "🔊 SPEAKING",
    }[state]


# ──────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────

def main():
    print_banner()

    state = State.LISTENING   # start directly in LISTENING after the greeting
    reply = ""
    lang  = "hi"

    # ── Opening greeting ──────────────────────────────────────
    speak(
        "Hello, mein RoboBot hoon"
        "mein aapki kese madad kar sakta hoon"
        "Krapya apna sawaal poochhiye",
        lang="hi",
    )

    try:
        while True:

            # ════════════════════════════════════════════════════
            #  IDLE — wait silently for wake word
            # ════════════════════════════════════════════════════
            if state == State.IDLE:
                print(f"\n{state_label(state)}  — say 'Hello' to activate...")

                audio = capture_speech(timeout=IDLE_POLL_TIMEOUT)

                if audio is None:
                    # Nobody spoke for IDLE_POLL_TIMEOUT — keep idling silently
                    continue

                print("🔍 Checking for wake word...")
                wake_text, _ = transcribe(audio)
                print(f"   Heard: {wake_text!r}")

                if is_wake_word(wake_text):
                    state = State.LISTENING
                    print("\n✅ Wake word detected!")
                    speak(
                        "Haan, mein sun raha hoon. Aap apna sawaal poochhiye.",
                        lang="hi",
                    )
                else:
                    print("   Not a wake word — staying idle.")

                continue

            # ════════════════════════════════════════════════════
            #  LISTENING — auto VAD; 10 s of silence → IDLE
            # ════════════════════════════════════════════════════
            if state == State.LISTENING:
                print(f"\n{state_label(state)}  "
                      f"— silence for {int(IDLE_TIMEOUT)}s → idle")

                audio = capture_speech(timeout=IDLE_TIMEOUT)

                if audio is None:
                    # 10 seconds of nothing — go idle
                    state = State.IDLE
                    print(f"\n⏱️  No speech for {int(IDLE_TIMEOUT)}s — going idle.")
                    speak(
                        "Mein abhi idle mode mein ja raha hoon. "
                        "Jab zaroorat ho, 'Hello' kahiye.",
                        lang="hi",
                    )
                    continue

                # ── Got speech — transcribe ────────────────────
                print("🔍 Transcribing...")
                user_text, lang = transcribe(audio)

                if not user_text:
                    print("⚠️  Could not understand — listening again.")
                    continue

                print(f"   You [{lang.upper()}] › {user_text}")

                # ── Get AI reply ───────────────────────────────
                print("🤔 Thinking...")
                reply = get_ai_reply(user_text, lang)
                print(f"   AI  [{lang.upper()}] › {reply}")

                state = State.SPEAKING
                continue

            # ════════════════════════════════════════════════════
            #  SPEAKING — play reply, then return to LISTENING
            # ════════════════════════════════════════════════════
            if state == State.SPEAKING:
                print(f"\n{state_label(state)}")
                speak(reply, lang)
                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        # speak(
        #     "Dhanyavaad. Aikcropolis College mein aapka swagat hai. Alvida!",
        #     lang="hi",
        # )


if __name__ == "__main__":
    main()