import os
import re
import numpy as np
import soundfile as sf
from TTS.api import TTS
import torch

# Workaround for lazy-loading ModuleNotFoundError in some environments
try:
    from transformers import GenerationMixin, GPT2PreTrainedModel
    print("Pre-loaded transformers in synthesize module.")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load XTTS v2 and move to GPU if available
_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[synthesize] Loading XTTS v2 on {_device.upper()}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to(_device)
print(f"[synthesize] XTTS v2 ready on {_device.upper()}.")

# ---------------------------------------------------------------------------
# Language / Accent mapping
# ---------------------------------------------------------------------------
# Maps accent label → XTTS language code.
# XTTS v2 natively supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, ko, hu
ACCENTS = {
    "neutral":    "en",
    "american":   "en",
    "british":    "en",
    "indian":     "en",       # XTTS doesn't have a separate Indian-English code
    "hindi":      "hi",       # Hindi synthesis (XTTS limited support; falls back gracefully)
    "spanish":    "es",
    "french":     "fr",
    "german":     "de",
    "italian":    "it",
    "portuguese": "pt",
    "russian":    "ru",
    "japanese":   "ja",
    "korean":     "ko",
    "chinese":    "zh-cn",
    "arabic":     "ar",
}

TONES = ["calm", "energetic", "dramatic", "whisper", "cheerful"]

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# XTTS v2 hard limit is ~400 tokens; 230 chars is a safe upper bound per chunk.
_MAX_CHUNK_CHARS = 230

# How much silence (seconds) to insert between chunks depending on punctuation context.
_SILENCE_PARAGRAPH = 0.35   # after \n\n or end of a long block
_SILENCE_SENTENCE   = 0.20  # after . ! ?
_SILENCE_CLAUSE     = 0.05  # after , ; :


def _split_into_chunks(text: str, max_chars: int = _MAX_CHUNK_CHARS):
    """
    Split text into sentence-based chunks within XTTS token limits.
    Returns list of (chunk_text, gap_after_seconds) tuples.
    """
    # Normalise paragraph breaks
    paragraphs = re.split(r'\n{2,}', text.strip())
    result = []

    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para.strip())
        current = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                # Long sentence: split on clause delimiters
                sub_parts = re.split(r'([,;:])\s*', sentence)
                # re.split with capture groups gives [text, delim, text, delim, ...]
                merged = []
                i = 0
                while i < len(sub_parts):
                    part = sub_parts[i]
                    delim = sub_parts[i + 1] if i + 1 < len(sub_parts) else ""
                    merged.append(part + delim)
                    i += 2 if delim else 1
                for part in merged:
                    if len(current) + len(part) + 2 <= max_chars:
                        current = (current + " " + part).strip()
                    else:
                        if current:
                            result.append((current, _SILENCE_CLAUSE))
                        current = part[:max_chars]
            elif len(current) + len(sentence) + 1 <= max_chars:
                current = (current + " " + sentence).strip()
            else:
                if current:
                    # Determine gap based on how the current chunk ends
                    gap = _SILENCE_SENTENCE if re.search(r'[.!?]$', current) else _SILENCE_CLAUSE
                    result.append((current, gap))
                current = sentence

        if current:
            result.append((current, _SILENCE_PARAGRAPH))  # paragraph-level gap

    return [(chunk, gap) for chunk, gap in result if chunk.strip()]


# ---------------------------------------------------------------------------
# Core synthesis helpers
# ---------------------------------------------------------------------------

def _get_speed_for_tone(tone: str, speed_override: float = None) -> float:
    """
    Map tone label to a synthesis speed multiplier, unless the caller supplies
    an explicit override.
    """
    if speed_override is not None:
        return float(speed_override)
    tone_speed = {
        "calm":      1.0,
        "energetic": 1.15,
        "dramatic":  0.90,
        "whisper":   0.85,
        "cheerful":  1.10,
    }
    return tone_speed.get(tone.lower(), 1.0)


def _synthesize_long_text(text: str, speaker_wav: str, language: str,
                           output_path: str, speed: float = 1.0):
    """
    Synthesise arbitrarily long text by splitting into sentence chunks and
    concatenating with prosody-aware gaps. Prevents XTTS 400-token AssertionError.
    """
    chunks = _split_into_chunks(text)
    if not chunks:
        raise ValueError("No speakable text found after chunking.")

    print(f"  [synthesize] Synthesising {len(chunks)} chunk(s) at speed={speed:.2f}...")
    audio_segments = []
    gap_sizes      = []
    sample_rate    = 24000  # XTTS v2 always outputs at 24 kHz

    for i, (chunk, gap_after) in enumerate(chunks):
        print(f"    Chunk {i+1}/{len(chunks)}: {chunk[:60]}{'...' if len(chunk) > 60 else ''}")
        audio_data = tts.tts(
            text=chunk,
            speaker_wav=speaker_wav,
            language=language,
            speed=speed,
        )
        audio_segments.append(np.array(audio_data, dtype=np.float32))
        gap_sizes.append(gap_after)

    # Concatenate with prosody-aware silence gaps
    parts = []
    for seg, gap in zip(audio_segments, gap_sizes):
        parts.append(seg)
        silence_samples = int(sample_rate * gap)
        if silence_samples > 0:
            parts.append(np.zeros(silence_samples, dtype=np.float32))

    # Drop trailing silence
    if parts and len(parts) > 1:
        parts = parts[:-1]

    final_audio = np.concatenate(parts)
    sf.write(output_path, final_audio, sample_rate)
    print(f"  [synthesize] Written to: {output_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize(text: str, speaker_wav: str, accent: str = "neutral",
               tone: str = "calm", output_path: str = None,
               language: str = None, speed: float = None):
    """
    Standard TTS clone: synthesise given text using speaker_wav as voice reference.

    Args:
        text:        The text to speak.
        speaker_wav: Path to the reference speaker audio clip.
        accent:      One of the ACCENTS keys (e.g. 'british', 'spanish').
        tone:        One of TONES ('calm', 'energetic', 'dramatic', 'whisper', 'cheerful').
        output_path: Where to save the output .wav (auto-generated if None).
        language:    Override language code directly (overrides accent mapping).
        speed:       Override speech rate 0.8–1.2 (overrides tone default).
    """
    from src.audio_processor import preprocess_speaker_wav, postprocess_output

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "xtts_output.wav")

    # Resolve language code
    if language:
        language_code = language
    else:
        language_code = ACCENTS.get(accent.lower(), "en")
        if accent.lower() == "indian":
            print("[synthesize] Note: Indian-English accent uses English model (no dedicated XTTS code).")

    if tone.lower() not in TONES:
        print(f"[synthesize] Tone '{tone}' not recognised — using 'calm'.")
        tone = "calm"

    effective_speed = _get_speed_for_tone(tone, speed)

    print(f"[synthesize] accent={accent}, tone={tone}, language={language_code}, "
          f"speed={effective_speed:.2f} → {output_path}")

    # Pre-process speaker reference for cleaner voice fingerprint
    clean_speaker_wav = preprocess_speaker_wav(speaker_wav)

    _synthesize_long_text(text, clean_speaker_wav, language_code, output_path, speed=effective_speed)

    # Post-process output (high-pass, limiter, normalise)
    postprocess_output(output_path)

    # Clean up temp preprocessed file if it was auto-generated
    if clean_speaker_wav != speaker_wav and os.path.exists(clean_speaker_wav):
        os.remove(clean_speaker_wav)

    print(f"[synthesize] Voice cloning complete: {output_path}")


def mimic_voice(performance_wav: str, target_voice_wav: str,
                output_path: str = None, speed: float = 1.0):
    """
    Smart Mimic pipeline: transcribe the performance audio, then re-synthesise
    those words using XTTS with the target person's voice.

    Args:
        performance_wav:  Path to the user's performance audio.
        target_voice_wav: Path to the target speaker's voice sample.
        output_path:      Where to save the result (auto-generated if None).
        speed:            Speech rate multiplier (0.8 – 1.2).

    Returns:
        (output_path, transcribed_text)
    """
    from src.transcribe import transcribe_audio
    from src.audio_processor import preprocess_speaker_wav, postprocess_output

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "smart_mimic_output.wav")

    # Step 1: Transcribe the performance audio
    print(f"[mimic] Transcribing performance: {performance_wav}")
    transcribed_text = transcribe_audio(performance_wav)

    if not transcribed_text or not transcribed_text.strip():
        raise ValueError(
            "Could not transcribe any speech from the performance audio. "
            "Ensure the audio is clear and contains audible speech."
        )
    print(f"[mimic] Transcribed ({len(transcribed_text)} chars): "
          f"{transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}")

    # Step 2: Pre-process target voice for a cleaner speaker fingerprint
    print(f"[mimic] Pre-processing target voice: {target_voice_wav}")
    clean_target_wav = preprocess_speaker_wav(target_voice_wav)

    # Step 3: Re-synthesise with target voice
    print(f"[mimic] Re-synthesising with target voice...")
    _synthesize_long_text(transcribed_text, clean_target_wav, "en", output_path, speed=speed)

    # Step 4: Post-process output
    postprocess_output(output_path)

    # Cleanup
    if clean_target_wav != target_voice_wav and os.path.exists(clean_target_wav):
        os.remove(clean_target_wav)

    print(f"[mimic] Done. Output: {output_path}")
    return output_path, transcribed_text