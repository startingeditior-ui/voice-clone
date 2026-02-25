from faster_whisper import WhisperModel
import torch
import os

# Set environment variable to suppress some logs if needed
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
# "small" is the recommended model for GPUs with 4GB VRAM (e.g. RTX 2050).
# large-v2 needs ~10GB and medium needs ~5GB — both exceed RTX 2050's limit.
# Set WHISPER_MODEL env var to override (e.g. "base", "small").
# ---------------------------------------------------------------------------
_DEFAULT_GPU_MODEL = os.environ.get("WHISPER_MODEL", "small")
_DEFAULT_CPU_MODEL = os.environ.get("WHISPER_MODEL_CPU", "small")

_on_cpu = False

try:
    if torch.cuda.is_available():
        print(f"CUDA detected – loading Whisper '{_DEFAULT_GPU_MODEL}' on GPU...")
        model = WhisperModel(_DEFAULT_GPU_MODEL, device="cuda", compute_type="float16")
        print(f"Whisper '{_DEFAULT_GPU_MODEL}' initialised on CUDA.")
    else:
        print(f"CUDA not available – loading Whisper '{_DEFAULT_CPU_MODEL}' on CPU (this may take a moment)...")
        model = WhisperModel(_DEFAULT_CPU_MODEL, device="cpu", compute_type="int8")
        _on_cpu = True
        print(f"Whisper '{_DEFAULT_CPU_MODEL}' initialised on CPU.")
except Exception as e:
    print(f"Failed to initialise primary Whisper model: {e}")
    print("Falling back to 'base' on CPU for transcription.")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    _on_cpu = True


def transcribe_audio(path: str) -> str:
    """
    Transcribe speech in an audio file and return the full text.

    Improvements over the original implementation:
    - Uses 'large-v2' (GPU) or 'medium' (CPU) model for higher accuracy.
    - VAD (voice activity detection) filter strips non-speech regions before
      transcription, reducing hallucinations and improving accuracy.
    - beam_size=5 uses a wider beam for more accurate decoding.
    """
    global model, _on_cpu

    transcribe_kwargs = dict(
        vad_filter=True,          # strip silence / non-speech before decoding
        vad_parameters=dict(
            min_silence_duration_ms=300,   # gaps shorter than this are kept
            speech_pad_ms=200,             # pad speech edges to avoid clipping
        ),
        beam_size=5,              # wider beam → slightly slower but more accurate
        best_of=5,                # keep the best of 5 candidates per segment
        temperature=0.0,          # greedy first; only fall back to sampling on failure
    )

    try:
        segments, info = model.transcribe(path, **transcribe_kwargs)
        detected_language = info.language if hasattr(info, "language") else "unknown"
        print(f"[transcribe] Detected language: {detected_language} "
              f"(confidence: {info.language_probability:.2f})" if hasattr(info, "language_probability") else
              f"[transcribe] Transcribing...")
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()

    except RuntimeError as e:
        # cuBLAS / CUDA DLL not found at runtime — hot-swap to CPU
        if not _on_cpu:
            print(f"CUDA transcription failed ({e}). Switching Whisper to CPU (small)...")
            model = WhisperModel("small", device="cpu", compute_type="int8")
            _on_cpu = True
            segments, info = model.transcribe(path, **transcribe_kwargs)
            text = " ".join(segment.text.strip() for segment in segments)
            return text.strip()
        raise
