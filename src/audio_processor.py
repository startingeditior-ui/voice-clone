"""
audio_processor.py — Pre- and post-processing utilities for voice cloning.

Pre-processing:
  - Noise reduction (spectral subtraction via noisereduce)
  - Silence trimming
  - Amplitude normalisation (target: -20 dBFS for speaker references)
  - Enforce minimum duration for XTTS speaker references

Post-processing:
  - High-pass filter (remove low-frequency rumble below 80 Hz)
  - Soft limiter (prevent clipping)
  - Amplitude normalisation (target: -3 dBFS peak for final output)
"""

import os
import tempfile
import warnings
import numpy as np
import soundfile as sf
import scipy.signal as signal

# Suppress noisy deprecation warnings from noisereduce / librosa
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPEAKER_REF_TARGET_DBFS = -20.0   # Target loudness for speaker references
OUTPUT_TARGET_PEAK_DBFS  = -3.0   # Target peak level for synthesised output
MIN_SPEAKER_DURATION_S   = 3.0    # Minimum speaker reference duration (XTTS needs ≥3s clear speech)
HIGHPASS_CUTOFF_HZ       = 80     # Remove rumble below this frequency
NOISE_REDUCE_STATIONARY  = True   # Use stationary noise reduction (fast, reliable)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(path: str):
    """Load a WAV/MP3 file and return (audio_array, sample_rate)."""
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:                # stereo → mono
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    return audio, sr


def _save_temp(audio: np.ndarray, sr: int, suffix: str = ".wav") -> str:
    """Save a numpy array to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    sf.write(tmp.name, audio, sr)
    return tmp.name


def _rms_dbfs(audio: np.ndarray) -> float:
    """Return RMS level in dBFS (full scale)."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return -100.0
    return 20.0 * np.log10(rms)


def _peak_dbfs(audio: np.ndarray) -> float:
    """Return peak level in dBFS."""
    peak = np.max(np.abs(audio))
    if peak < 1e-10:
        return -100.0
    return 20.0 * np.log10(peak)


def _normalize_rms(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale audio so its RMS matches target_dbfs."""
    current = _rms_dbfs(audio)
    if current <= -99:
        return audio
    gain = 10 ** ((target_dbfs - current) / 20.0)
    return np.clip(audio * gain, -1.0, 1.0)


def _normalize_peak(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale audio so its peak matches target_dbfs."""
    current = _peak_dbfs(audio)
    if current <= -99:
        return audio
    gain = 10 ** ((target_dbfs - current) / 20.0)
    return np.clip(audio * gain, -1.0, 1.0)


def _highpass_filter(audio: np.ndarray, sr: int, cutoff: int = HIGHPASS_CUTOFF_HZ) -> np.ndarray:
    """Apply a Butterworth high-pass filter to remove low-frequency rumble."""
    nyq = sr / 2.0
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        return audio
    b, a = signal.butter(4, normal_cutoff, btype="high", analog=False)
    return signal.filtfilt(b, a, audio).astype(np.float32)


def _trim_silence(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """Remove leading/trailing silence using a simple energy threshold."""
    frame_len = int(sr * 0.025)       # 25ms frames
    hop_len   = int(sr * 0.010)       # 10ms hop
    threshold = 10 ** (-top_db / 20.0)

    # Compute short-time energy
    frames = []
    for start in range(0, len(audio) - frame_len, hop_len):
        frame = audio[start : start + frame_len]
        frames.append(np.sqrt(np.mean(frame ** 2)))

    if not frames:
        return audio

    energy = np.array(frames)
    voiced = energy > threshold

    if not voiced.any():
        return audio                   # all silence — return as-is

    first = np.argmax(voiced) * hop_len
    last  = (len(voiced) - np.argmax(voiced[::-1]) - 1) * hop_len + frame_len
    last  = min(last, len(audio))
    return audio[first:last]


def _denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply spectral-subtraction noise reduction."""
    try:
        import noisereduce as nr
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=NOISE_REDUCE_STATIONARY,
            prop_decrease=0.8,    # how aggressively to suppress noise
        )
        return reduced.astype(np.float32)
    except Exception as e:
        print(f"[audio_processor] Noise reduction skipped: {e}")
        return audio


def _soft_limit(audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Apply a soft knee limiter to prevent harsh clipping."""
    knee = np.where(
        np.abs(audio) < threshold,
        audio,
        np.sign(audio) * (threshold + (1.0 - threshold) *
                          np.tanh((np.abs(audio) - threshold) / (1.0 - threshold)))
    )
    return knee.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_speaker_wav(input_path: str, output_path: str = None) -> str:
    """
    Pre-process a speaker reference audio file for XTTS voice cloning.

    Steps applied:
      1. Convert to mono / float32
      2. Trim leading / trailing silence
      3. Spectral noise reduction
      4. High-pass filter (remove rumble)
      5. RMS normalise to -20 dBFS

    Args:
        input_path:  Path to the original speaker audio file.
        output_path: Where to save the processed file.
                     Defaults to a temp file if None.

    Returns:
        Path to the pre-processed WAV file.
    """
    print(f"[audio_processor] Pre-processing speaker reference: {input_path}")
    audio, sr = _load(input_path)

    # 1. Trim silence
    audio = _trim_silence(audio, sr)

    # 2. Check minimum duration
    duration = len(audio) / sr
    if duration < MIN_SPEAKER_DURATION_S:
        print(f"[audio_processor] WARNING: Speaker reference is only {duration:.1f}s "
              f"(XTTS works best with ≥{MIN_SPEAKER_DURATION_S}s of clear speech). "
              f"Using anyway.")

    # 3. Noise reduction
    audio = _denoise(audio, sr)

    # 4. High-pass filter
    audio = _highpass_filter(audio, sr)

    # 5. RMS normalise
    audio = _normalize_rms(audio, SPEAKER_REF_TARGET_DBFS)

    if output_path is None:
        output_path = _save_temp(audio, sr)
    else:
        sf.write(output_path, audio, sr)

    print(f"[audio_processor] Speaker reference ready: {output_path} "
          f"({len(audio)/sr:.1f}s, RMS={_rms_dbfs(audio):.1f} dBFS)")
    return output_path


def postprocess_output(audio_path: str) -> str:
    """
    Post-process a synthesised output WAV in-place.

    Steps applied:
      1. High-pass filter (remove low-frequency rumble)
      2. Soft limiter (prevent clipping)
      3. Peak normalise to -3 dBFS

    Args:
        audio_path: Path to the synthesised WAV file (modified in-place).

    Returns:
        Same path (modified in-place).
    """
    print(f"[audio_processor] Post-processing output: {audio_path}")
    audio, sr = _load(audio_path)

    # 1. High-pass filter
    audio = _highpass_filter(audio, sr)

    # 2. Soft limiter
    audio = _soft_limit(audio)

    # 3. Peak normalise
    audio = _normalize_peak(audio, OUTPUT_TARGET_PEAK_DBFS)

    sf.write(audio_path, audio, sr)
    print(f"[audio_processor] Post-processing done. "
          f"Peak={_peak_dbfs(audio):.1f} dBFS, RMS={_rms_dbfs(audio):.1f} dBFS")
    return audio_path


def get_duration(audio_path: str) -> float:
    """Return the duration of an audio file in seconds."""
    audio, sr = _load(audio_path)
    return len(audio) / sr
