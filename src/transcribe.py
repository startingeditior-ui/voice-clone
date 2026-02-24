from faster_whisper import WhisperModel
import torch
import os

# Set environment variable to suppress some logs if needed
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Device detection for transcription
_on_cpu = False

try:
    if torch.cuda.is_available():
        print("CUDA detected, attempting to use GPU for transcription...")
        model = WhisperModel("base", device="cuda", compute_type="float16")
        print("Whisper initialized with CUDA.")
    else:
        print("CUDA not available, using CPU for transcription.")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        _on_cpu = True
except Exception as e:
    print(f"Failed to initialize Whisper with CUDA: {e}")
    print("Falling back to CPU for transcription.")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    _on_cpu = True

def transcribe_audio(path):
    global model, _on_cpu
    try:
        segments, info = model.transcribe(path)
        text = ""
        for segment in segments:
            text += segment.text + " "
        return text.strip()
    except RuntimeError as e:
        # cuBLAS / CUDA DLL not found at runtime — hot-swap to CPU
        if not _on_cpu:
            print(f"CUDA transcription failed ({e}). Switching Whisper to CPU...")
            model = WhisperModel("base", device="cpu", compute_type="int8")
            _on_cpu = True
            segments, info = model.transcribe(path)
            text = ""
            for segment in segments:
                text += segment.text + " "
            return text.strip()
        raise
