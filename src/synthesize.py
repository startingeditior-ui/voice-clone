import os
import re
import numpy as np
import soundfile as sf
from TTS.api import TTS

# Workaround for lazy-loading ModuleNotFoundError in some environments
try:
    from transformers import GenerationMixin, GPT2PreTrainedModel
    print("Pre-loaded transformers in synthesize module.")
except ImportError:
    pass

# Resolve output directory relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

ACCENTS = {
    "neutral": "en",
    "american": "en",
    "british": "en",
    "indian": "en"
}

TONES = ["calm", "energetic", "dramatic"]

# XTTS v2 hard limit is ~400 tokens; 230 chars is a safe chunk size
_MAX_CHUNK_CHARS = 230


def _split_into_chunks(text, max_chars=_MAX_CHUNK_CHARS):
    """Split text into sentence-based chunks that stay within XTTS token limits."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            # Very long single sentence: split on commas / semicolons
            sub_parts = re.split(r'[,;]\s*', sentence)
            for part in sub_parts:
                if len(current) + len(part) + 2 <= max_chars:
                    current = (current + " " + part).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = part[:max_chars]
        elif len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def _synthesize_long_text(text, speaker_wav, language, output_path):
    """
    Synthesize arbitrarily long text by splitting into sentence chunks and
    concatenating the resulting audio. Prevents XTTS 400-token AssertionError.
    """
    chunks = _split_into_chunks(text)
    if not chunks:
        raise ValueError("No speakable text found after chunking.")

    print(f"  Synthesizing {len(chunks)} chunk(s)...")
    audio_segments = []
    sample_rate = 24000  # XTTS v2 always outputs at 24 kHz

    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i+1}/{len(chunks)}: {chunk[:60]}{'...' if len(chunk)>60 else ''}")
        audio_data = tts.tts(
            text=chunk,
            speaker_wav=speaker_wav,
            language=language
        )
        audio_segments.append(np.array(audio_data, dtype=np.float32))

    # Join chunks with a short silence (0.2 s) between them
    silence = np.zeros(int(sample_rate * 0.2), dtype=np.float32)
    parts = []
    for seg in audio_segments:
        parts.append(seg)
        parts.append(silence)
    final_audio = np.concatenate(parts[:-1])  # drop trailing silence

    sf.write(output_path, final_audio, sample_rate)
    print(f"Synthesis complete: {output_path}")


def synthesize(text, speaker_wav, accent="neutral", tone="calm", output_path=None):
    """Standard TTS: synthesize given text using speaker_wav as voice reference."""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "xtts_output.wav")

    language_code = ACCENTS.get(accent.lower(), "en")
    if accent.lower() == "indian":
        print("Warning: Indian English accent not natively supported, using neutral English.")

    if tone.lower() not in TONES:
        print(f"Warning: Tone '{tone}' not recognized, using 'calm'.")
        tone = "calm"

    print(f"Generating voice with accent={accent}, tone={tone} to {output_path}...")
    _synthesize_long_text(text, speaker_wav, language_code, output_path)
    print(f"Voice generated successfully at {output_path}")


def mimic_voice(performance_wav, target_voice_wav, output_path=None):
    """
    Smart Mimic pipeline: Transcribes the user's performance audio, then
    re-synthesizes those exact words using XTTS with the target person's
    voice as the speaker reference.

    Args:
        performance_wav: Path to the user's own recorded performance
        target_voice_wav: Path to the target person's voice to mimic
        output_path: Where to save the output .wav (auto-generated if None)

    Returns:
        (output_path, transcribed_text)
    """
    from src.transcribe import transcribe_audio

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "smart_mimic_output.wav")

    # Step 1: Transcribe the performance audio
    print(f"[Mimic] Transcribing performance: {performance_wav}")
    transcribed_text = transcribe_audio(performance_wav)

    if not transcribed_text or not transcribed_text.strip():
        raise ValueError(
            "Could not transcribe any speech from the performance audio. "
            "Ensure the audio is clear and contains audible speech."
        )
    snippet = transcribed_text[:100]
    print(f"[Mimic] Transcribed: {snippet}{'...' if len(transcribed_text) > 100 else ''}")

    # Step 2: Re-synthesize with target voice
    print(f"[Mimic] Re-synthesizing with target voice: {target_voice_wav}")
    _synthesize_long_text(transcribed_text, target_voice_wav, "en", output_path)

    print(f"[Mimic] Done. Output: {output_path}")
    return output_path, transcribed_text