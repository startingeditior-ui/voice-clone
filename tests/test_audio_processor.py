"""
tests/test_audio_processor.py — Tests for the audio preprocessing/postprocessing module.

Generates a synthetic noisy WAV in memory, runs it through preprocess_speaker_wav()
and postprocess_output(), then asserts the output meets the expected quality criteria.
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf

# Ensure the project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio_processor import (
    preprocess_speaker_wav,
    postprocess_output,
    get_duration,
    _rms_dbfs,
    _peak_dbfs,
    SPEAKER_REF_TARGET_DBFS,
    OUTPUT_TARGET_PEAK_DBFS,
)


def _make_noisy_wav(duration_s: float = 5.0, sample_rate: int = 22050) -> str:
    """Create a temporary WAV file with a 440 Hz tone mixed with white noise."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # 440 Hz sine wave at 80% amplitude + white noise at 10% amplitude
    audio = 0.8 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio, sample_rate)
    tmp.close()
    return tmp.name


def test_preprocess_speaker_wav():
    """preprocess_speaker_wav should return a valid WAV with reasonable RMS level."""
    noisy_wav = _make_noisy_wav(duration_s=5.0)
    out_tmp = tempfile.mktemp(suffix=".wav")

    try:
        result_path = preprocess_speaker_wav(noisy_wav, output_path=out_tmp)

        # File should exist
        assert os.path.exists(result_path), "Output file does not exist"

        # Duration should be preserved (within a 1s margin for trim)
        duration = get_duration(result_path)
        assert duration >= 1.0, f"Duration too short after preprocessing: {duration:.2f}s"

        # RMS should be within ±6 dB of target (normalisation may not be perfect)
        audio, _ = sf.read(result_path)
        rms = _rms_dbfs(np.array(audio, dtype=np.float32))
        assert SPEAKER_REF_TARGET_DBFS - 6 <= rms <= SPEAKER_REF_TARGET_DBFS + 6, (
            f"RMS {rms:.1f} dBFS is too far from target {SPEAKER_REF_TARGET_DBFS} dBFS"
        )

        print(f"✅ preprocess_speaker_wav OK — duration={duration:.1f}s, RMS={rms:.1f} dBFS")

    finally:
        for p in [noisy_wav, out_tmp]:
            if os.path.exists(p):
                os.remove(p)


def test_postprocess_output():
    """postprocess_output should produce an output with peak ≤ OUTPUT_TARGET_PEAK_DBFS."""
    noisy_wav = _make_noisy_wav(duration_s=3.0)

    try:
        postprocess_output(noisy_wav)

        audio, _ = sf.read(noisy_wav)
        peak = _peak_dbfs(np.array(audio, dtype=np.float32))

        # Peak should be at or below the target (with 1 dB tolerance for floating point)
        assert peak <= OUTPUT_TARGET_PEAK_DBFS + 1.0, (
            f"Peak {peak:.1f} dBFS exceeds target {OUTPUT_TARGET_PEAK_DBFS} dBFS"
        )

        print(f"✅ postprocess_output OK — peak={peak:.1f} dBFS")

    finally:
        if os.path.exists(noisy_wav):
            os.remove(noisy_wav)


def test_get_duration():
    """get_duration should return correct duration for a known-length WAV."""
    wav = _make_noisy_wav(duration_s=4.0, sample_rate=16000)
    try:
        dur = get_duration(wav)
        assert 3.9 <= dur <= 4.1, f"Duration {dur:.2f}s not close to expected 4.0s"
        print(f"✅ get_duration OK — {dur:.2f}s")
    finally:
        if os.path.exists(wav):
            os.remove(wav)


if __name__ == "__main__":
    print("Running audio_processor tests...\n")
    test_preprocess_speaker_wav()
    test_postprocess_output()
    test_get_duration()
    print("\n✅ All audio_processor tests passed!")
