"""Bass stem separation via Meta Demucs (htdemucs) with SciPy bandpass fallback."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

from bass_mcp.security import SecurityValidationError, run_safe_command

VALID_SEPARATION_METHODS = frozenset({"auto", "demucs", "lowpass", "none"})


def has_demucs() -> bool:
    """Return True if the demucs package is installed in the current Python environment."""
    return importlib.util.find_spec("demucs") is not None


def _bandpass_bass_wav(audio_wav: Path, out_wav: Path) -> Path:
    """Isolate bass frequencies (30 Hz - 260 Hz) using a 4th-order Butterworth bandpass filter."""
    sr, data = wavfile.read(str(audio_wav))
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        signal = data.astype(np.float32) / max(max_val, 1.0)
    else:
        signal = data.astype(np.float32)

    if len(signal) < 32:
        raise ValueError("Audio signal is too short for bass frequency filtering.")

    nyquist = sr * 0.5
    low_hz = max(25.0, min(30.0, nyquist * 0.2))
    high_hz = min(260.0, nyquist * 0.85)
    if low_hz >= high_hz:
        filtered = signal
    else:
        sos = butter(4, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
        filtered = sosfiltfilt(sos, signal).astype(np.float32)

    peak = float(np.max(np.abs(filtered))) if len(filtered) else 0.0
    if peak > 1e-6:
        filtered = filtered * min(1.0, 0.9 / peak)

    pcm16 = np.clip(filtered * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(str(out_wav), int(sr), pcm16)
    return out_wav


def _demucs_bass_wav(audio_wav: Path, job_dir: Path) -> Path:
    """Run Demucs (htdemucs --two-stems=bass) safely in a subprocess."""
    demucs_out = job_dir / "demucs_stems"
    demucs_out.mkdir(parents=True, exist_ok=True)
    run_safe_command(
        [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            "--two-stems=bass",
            "-o",
            str(demucs_out),
            "--",
            str(audio_wav),
        ],
        timeout=600,
        cwd=job_dir,
    )
    expected = demucs_out / "htdemucs" / audio_wav.stem / "bass.wav"
    if expected.is_file():
        return expected
    matches = sorted(demucs_out.glob("**/bass.wav"))
    if not matches:
        raise RuntimeError("Demucs finished without producing a bass.wav stem.")
    return matches[0]


def separate_bass_stem(audio_wav: Path, job_dir: Path, method: str = "auto") -> dict:
    """Extract the bass track from a normalized WAV file.

    Returns ``{"bass_wav": Path, "engine": str}``.
    """
    normalized_method = method.strip().lower()
    if normalized_method not in VALID_SEPARATION_METHODS:
        raise SecurityValidationError(
            f"Unsupported separation method '{method}'. Expected one of: {sorted(VALID_SEPARATION_METHODS)}"
        )

    if normalized_method == "none":
        return {"bass_wav": audio_wav, "engine": "none"}

    if normalized_method == "demucs" or (normalized_method == "auto" and has_demucs()):
        if not has_demucs():
            raise RuntimeError("demucs is not installed. Install optional dependencies with: pip install -e '.[ml]'")
        bass_path = _demucs_bass_wav(audio_wav, job_dir)
        return {"bass_wav": bass_path, "engine": "demucs:htdemucs"}

    out_wav = job_dir / "bass_isolated.wav"
    _bandpass_bass_wav(audio_wav, out_wav)
    return {"bass_wav": out_wav, "engine": "scipy:bandpass_30_260hz"}
