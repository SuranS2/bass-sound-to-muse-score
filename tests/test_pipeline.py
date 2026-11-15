"""End-to-end pipeline test with a synthesized 4-note electric bass WAV fixture."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from bass_mcp.metadata import _fetch_allowlisted_json, parse_song_artist
from bass_mcp.security import SecurityValidationError
from bass_mcp.server import (
    audio_to_bass_score,
    check_environment_status,
    separate_bass_track,
    transcribe_bass_pipeline,
)
from bass_mcp.transcriber import has_librosa, midi_to_hz, transcribe_bass_notes


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


def test_parse_song_artist_and_lookup_security() -> None:
    meta = parse_song_artist(
        "22. 은혼 17기 ED, SPYAIR - Some Like It Hot 베이스 커버 【★★★☆☆】 | BassGuitar tutorial | 악보[TAB]"
    )
    assert meta["artist"] == "SPYAIR"
    assert meta["song_title"] == "Some Like It Hot"
    assert meta["display_name"] == "SPYAIR - Some Like It Hot"
    assert meta["web_search_query"] == "SPYAIR Some Like It Hot BPM"

    with pytest.raises(SecurityValidationError):
        _fetch_allowlisted_json("http://api.deezer.com/search?q=test")
    with pytest.raises(SecurityValidationError):
        _fetch_allowlisted_json("https://127.0.0.1/metadata")


@pytest.mark.asyncio
async def test_pipeline_pauses_for_bpm_confirmation_then_finalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_root = tmp_path / "mcp_output"
    monkeypatch.setenv("BASS_MCP_OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(
        "bass_mcp.server.lookup_original_song_bpm",
        lambda song_title, artist="": {
            "query": f"{artist} {song_title}".strip(),
            "web_search_query": "SPYAIR Some Like It Hot BPM",
            "found": True,
            "bpm": 113.0,
            "source": "mock:test",
            "canonical_artist": "SPYAIR",
            "canonical_title": "Some Like It Hot!!",
        },
    )

    input_wav = _synthesize_bass_riff_wav(tmp_path / "SPYAIR - Some Like It Hot [Bass Cover].wav")
    pending = await transcribe_bass_pipeline(
        source=str(input_wav),
        bpm=None,
        separation_method="lowpass",
        send_to_musescore=False,
    )
    assert pending["status"] == "needs_bpm_confirmation"
    assert pending["requires_user_confirmation"] is True
    assert pending["recommended_bpm"] == 113.0
    assert pending["recommended_time_signature"] == "4/4"
    assert pending["song_metadata"]["canonical_artist"] == "SPYAIR"
    assert "SPYAIR" in pending["confirmation_prompt_ko"]
    assert Path(pending["files"]["bass_wav"]).is_file()

    # Simulate user confirming/correcting BPM to 113.0 and selecting custom time signature 3/4.
    final = audio_to_bass_score(
        audio_path=pending["files"]["bass_wav"],
        title=pending["title"],
        bpm=113.0,
        time_signature="3/4",
    )
    assert final["status"] == "ok"
    assert final["bpm"] == 113.0
    assert final["time_signature"] == "3/4"
    final_xml = ET.parse(Path(final["files"]["musicxml"])).getroot()
    assert final_xml.findtext(".//attributes/time/beats") == "3"
    assert final_xml.findtext(".//attributes/time/beat-type") == "4"


def test_repeated_same_pitch_notes_split_cleanly(tmp_path: Path) -> None:
    if not has_librosa():
        pytest.skip("librosa not installed")

    sr = 22050
    f0 = midi_to_hz(36)  # C2
    note_dur = 0.25  # 8th notes at 120 BPM
    chunks = []
    for _ in range(6):
        t = np.linspace(0.0, note_dur, int(sr * note_dur), endpoint=False)
        env = np.minimum(1.0, t / 0.01) * np.exp(-3.5 * t)
        wave = (0.75 * np.sin(2.0 * np.pi * f0 * t) + 0.25 * np.sin(4.0 * np.pi * f0 * t)) * env
        chunks.append(wave)
    audio = np.concatenate(chunks)
    wav_path = tmp_path / "repeated_c2.wav"
    wavfile.write(str(wav_path), sr, np.clip(audio * 24000.0, -32768, 32767).astype(np.int16))

    res = transcribe_bass_notes(wav_path, min_midi=28)
    assert res["engine"] == "librosa:pyin"
    assert len(res["notes"]) >= 5


def test_two_tap_16th_double_hits_preserved() -> None:
    from bass_mcp.transcriber import BassNote, _clean_transcribed_notes

    # Two 16th-note double taps (~133ms IOI at 113 BPM) must NOT merge, while 55ms jitter merges.
    notes = [
        BassNote(32, "G#1", 1.000, 0.130, 86),
        BassNote(32, "G#1", 1.132, 0.140, 84),
        BassNote(31, "G1", 2.000, 0.055, 76),
        BassNote(31, "G1", 2.055, 0.150, 88),
    ]
    cleaned = _clean_transcribed_notes(notes, min_velocity=62)
    assert len(cleaned) == 3
    assert [round(n.onset_sec, 3) for n in cleaned] == [1.0, 1.132, 2.0]


def test_parse_lrc_lyrics_and_semitone_wobble_suppression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bass_mcp.metadata import parse_lrc_lyrics
    from bass_mcp.transcriber import _segments_from_frame_pitches

    lines = parse_lrc_lyrics("[00:10.88] I am standing in the line\n[00:16.34] Shining like the sun\n")
    assert len(lines) == 2
    assert lines[0].onset_sec == 10.88
    assert lines[0].text == "I am standing in the line"

    # Simulate an alternating A1(33) bass line with a brief 5-frame G#1(32) attack dip.
    midi = np.array([33] * 14 + [32] * 4 + [33] * 14, dtype=np.int32)
    rms = np.full(len(midi), 0.5, dtype=np.float32)
    segs = _segments_from_frame_pitches(
        midi_frames=midi,
        rms_frames=rms,
        hop_sec=256 / 22050.0,
        min_midi=28,
        max_midi=67,
        min_note_len_sec=0.06,
    )
    assert all(n.pitch_midi == 33 for n in segs)

    out_root = tmp_path / "mcp_output"
    monkeypatch.setenv("BASS_MCP_OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(
        "bass_mcp.server.lookup_song_lyrics",
        lambda song_title, artist="", duration_sec=None: {
            "found": True,
            "synced_lines": lines,
            "plain_lines": [],
            "canonical_artist": "ELLEGARDEN",
            "canonical_title": "Marry Me",
            "source": "mock:lrclib",
        },
    )
    wav_path = _synthesize_bass_riff_wav(tmp_path / "raw.wav")
    res = audio_to_bass_score(audio_path=str(wav_path), title="Marry Me - ELLEGARDEN", bpm=120.0)
    assert Path(res["job_dir"]).name.startswith("Marry_Me-ELLEGARDEN-")
    assert res["lyrics_count"] == 2
