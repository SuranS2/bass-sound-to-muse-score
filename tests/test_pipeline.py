"""End-to-end pipeline test with a synthesized 4-note electric bass WAV fixture."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from bass_mcp.security import SecurityValidationError
from bass_mcp.server import (
    audio_to_bass_score,
    check_environment_status,
    separate_bass_track,
    transcribe_bass_pipeline,
)
from bass_mcp.transcriber import midi_to_hz


def _synthesize_bass_riff_wav(path: Path, sr: int = 44100) -> Path:
    """Synthesize a clean 4-beat bass riff: E1 (28), G1 (31), A1 (33), B1 (35) at 120 BPM."""
    notes_midi = [28, 31, 33, 35]
    note_dur = 0.5  # 120 BPM quarter notes -> 2.0s total
    segments: list[np.ndarray] = []

    for midi in notes_midi:
        f0 = midi_to_hz(midi)
        t = np.linspace(0.0, note_dur, int(sr * note_dur), endpoint=False)
        env = np.minimum(1.0, t / 0.015) * np.exp(-2.2 * t)
        # Fundamental + mild 2nd & 3rd harmonics typical of plucked bass
        wave = (
            0.72 * np.sin(2.0 * np.pi * f0 * t)
            + 0.22 * np.sin(2.0 * np.pi * 2.0 * f0 * t)
            + 0.06 * np.sin(2.0 * np.pi * 3.0 * f0 * t)
        ) * env
        segments.append(wave)

    audio = np.concatenate(segments)
    pcm16 = np.clip(audio * 24000.0, -32768, 32767).astype(np.int16)
    wavfile.write(str(path), sr, pcm16)
    return path


@pytest.mark.asyncio
async def test_end_to_end_bass_pipeline_on_synthetic_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_root = tmp_path / "mcp_output"
    monkeypatch.setenv("BASS_MCP_OUTPUT_DIR", str(out_root))

    input_wav = _synthesize_bass_riff_wav(tmp_path / "riff.wav")

    result = await transcribe_bass_pipeline(
        source=str(input_wav),
        title="Synthetic Bass Test",
        bpm=120.0,
        separation_method="lowpass",
        string_count=4,
        send_to_musescore=False,
    )

    assert result["status"] == "ok"
    assert result["source_type"] == "local_file"
    assert result["bpm"] == 120.0
    assert result["note_count"] >= 4

    detected_midis = [n["midi"] for n in result["notes_preview"]]
    for expected in (28, 31, 33, 35):
        assert expected in detected_midis

    midi_file = Path(result["files"]["midi"])
    musicxml_file = Path(result["files"]["musicxml"])
    assert midi_file.is_file() and midi_file.read_bytes().startswith(b"MThd")
    assert musicxml_file.is_file()
    ET.parse(musicxml_file)

    assert "E|" in result["ascii_tab"]
    assert len(result["chords"]) >= 1


def test_separate_and_audio_to_score_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_root = tmp_path / "mcp_output"
    monkeypatch.setenv("BASS_MCP_OUTPUT_DIR", str(out_root))

    input_wav = _synthesize_bass_riff_wav(tmp_path / "riff2.wav")

    sep = separate_bass_track(source=str(input_wav), method="lowpass")
    assert sep["status"] == "ok"
    assert Path(sep["bass_wav"]).is_file()

    scored = audio_to_bass_score(audio_path=sep["bass_wav"], title="Isolated Bass", bpm=120.0)
    assert scored["status"] == "ok"
    assert scored["note_count"] >= 4
    assert Path(scored["files"]["musicxml"]).is_file()

    with pytest.raises(SecurityValidationError):
        audio_to_bass_score(audio_path="https://www.youtube.com/watch?v=dQw4w9WgXcQ")


def test_check_environment_status() -> None:
    status = check_environment_status()
    assert status["status"] == "ok"
    projects = {item["project"] for item in status["attributions"]}
    assert "spotify/basic-pitch" in projects
    assert "facebookresearch/demucs" in projects
    assert "ghchen99/mcp-musescore" in projects
