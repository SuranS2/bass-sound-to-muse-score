"""FastMCP server exposing bass stem separation, transcription, TAB/chord scoring, and MuseScore control."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from bass_mcp.downloader import acquire_audio
from bass_mcp.musescore_bridge import MuseScoreBridge
from bass_mcp.score_builder import (
    TabbedBassNote,
    assign_bass_tab_positions,
    convert_with_musescore_cli,
    find_musescore_binary,
    infer_measure_chords,
    render_ascii_bass_tab,
    write_midi_file,
    write_musicxml_score,
)
from bass_mcp.security import (
    SecurityValidationError,
    parse_allowed_dirs,
    is_uri_input,
    prepare_job_output_dir,
    validate_input_audio_path,
    validate_input_file_path,
)
from bass_mcp.separator import has_demucs, separate_bass_stem
from bass_mcp.transcriber import (
    DEFAULT_BASS_MIN_MIDI,
    FIVE_STRING_MIN_MIDI,
    has_basic_pitch,
    has_librosa,
    midi_to_note_name,
    transcribe_bass_notes,
)

mcp = FastMCP("bass-sound-to-musescore")

_XML_DOCTYPE_RE = re.compile(rb"<!(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_MAX_PUSH_PAYLOAD_BYTES = 2 * 1024 * 1024


def _get_output_root() -> Path:
    configured = os.environ.get("BASS_MCP_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path.cwd() / "output"


def _sanitize_title(title: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", str(title or "")).strip()
    return (cleaned[:120] or "Bass Score")


def _min_midi_for_strings(string_count: int) -> int:
    if string_count == 4:
        return DEFAULT_BASS_MIN_MIDI
    if string_count == 5:
        return FIVE_STRING_MIN_MIDI
    raise SecurityValidationError("string_count must be 4 or 5.")


def _build_score_artifacts(
    notes: list[Any],
    job_dir: Path,
    title: str,
    bpm: float,
    string_count: int,
) -> dict[str, Any]:
    tabbed = assign_bass_tab_positions(notes, string_count=string_count)
    chords = infer_measure_chords(tabbed, bpm=bpm)

    midi_path = write_midi_file(tabbed, job_dir / "bass_line.mid", bpm=bpm)
    musicxml_path = write_musicxml_score(
        tabbed,
        chords,
        job_dir / "bass_score.musicxml",
        title=title,
        bpm=bpm,
        string_count=string_count,
    )
    ascii_tab = render_ascii_bass_tab(tabbed, chords, bpm=bpm, string_count=string_count)
    mscz_path = convert_with_musescore_cli(musicxml_path, output_format="mscz")

    return {
        "tabbed_notes": tabbed,
        "chords": [{"measure": c.measure_num, "symbol": c.symbol} for c in chords],
        "midi_path": midi_path,
        "musicxml_path": musicxml_path,
        "mscz_path": mscz_path,
        "ascii_tab": ascii_tab,
    }


@mcp.tool()
async def transcribe_bass_pipeline(
    source: str,
    title: str = "Bass Score",
    bpm: float | None = None,
    separation_method: str = "auto",
    string_count: int = 4,
    send_to_musescore: bool = False,
) -> dict[str, Any]:
    """Convert a YouTube URL or local audio file into Bass MusicXML, 4/5-string TAB, chords, and MIDI."""
    safe_title = _sanitize_title(title)
    min_midi = _min_midi_for_strings(string_count)
    job_dir = prepare_job_output_dir(_get_output_root())

    acq = acquire_audio(source=source, job_dir=job_dir)
    sep = separate_bass_stem(acq["wav_path"], job_dir=job_dir, method=separation_method)
    trans = transcribe_bass_notes(sep["bass_wav"], min_midi=min_midi, bpm_hint=bpm)

    effective_bpm = float(bpm) if bpm and bpm > 0 else float(trans["estimated_bpm"])
    scored = _build_score_artifacts(
        notes=trans["notes"],
        job_dir=job_dir,
        title=safe_title,
        bpm=effective_bpm,
        string_count=string_count,
    )

    push_result: dict[str, Any] | None = None
    if send_to_musescore:
        bridge = MuseScoreBridge(host="127.0.0.1", port=8765)
        push_result = await bridge.push_transcribed_notes(scored["tabbed_notes"], title=safe_title)

    tabbed: list[TabbedBassNote] = scored["tabbed_notes"]
    return {
        "status": "ok",
        "title": safe_title,
        "job_dir": str(job_dir),
        "source_type": acq["source_type"],
        "source_ref": acq["source_ref"],
        "duration_sec": acq["duration_sec"],
        "bpm": effective_bpm,
        "string_count": string_count,
        "engines": {
            "separator": sep["engine"],
            "transcriber": trans["engine"],
        },
        "files": {
            "source_wav": str(acq["wav_path"]),
            "bass_wav": str(sep["bass_wav"]),
            "midi": str(scored["midi_path"]),
            "musicxml": str(scored["musicxml_path"]),
            "mscz": str(scored["mscz_path"]) if scored["mscz_path"] else None,
        },
        "note_count": len(tabbed),
        "chords": scored["chords"],
        "ascii_tab": scored["ascii_tab"],
        "notes_preview": [
            {
                "note": n.note_name,
                "midi": n.pitch_midi,
                "onset_sec": n.onset_sec,
                "duration_sec": n.duration_sec,
                "string": n.string_num,
                "fret": n.fret,
            }
            for n in tabbed[:32]
        ],
        "musescore_push": push_result,
    }


@mcp.tool()
def separate_bass_track(source: str, method: str = "auto") -> dict[str, Any]:
    """Download or load audio and isolate only the bass stem WAV."""
    job_dir = prepare_job_output_dir(_get_output_root())
    acq = acquire_audio(source=source, job_dir=job_dir)
    sep = separate_bass_stem(acq["wav_path"], job_dir=job_dir, method=method)
    return {
        "status": "ok",
        "job_dir": str(job_dir),
        "source_type": acq["source_type"],
        "source_ref": acq["source_ref"],
        "duration_sec": acq["duration_sec"],
        "engine": sep["engine"],
        "source_wav": str(acq["wav_path"]),
        "bass_wav": str(sep["bass_wav"]),
    }


@mcp.tool()
def audio_to_bass_score(
    audio_path: str,
    title: str = "Bass Score",
    bpm: float | None = None,
    string_count: int = 4,
) -> dict[str, Any]:
    """Convert an already-isolated local bass audio file into MusicXML, TAB, chords, and MIDI."""
    if is_uri_input(audio_path):
        raise SecurityValidationError("audio_to_bass_score accepts local file paths only, not URLs.")
    validate_input_audio_path(audio_path)

    safe_title = _sanitize_title(title)
    min_midi = _min_midi_for_strings(string_count)
    job_dir = prepare_job_output_dir(_get_output_root())

    acq = acquire_audio(source=audio_path, job_dir=job_dir)
    trans = transcribe_bass_notes(acq["wav_path"], min_midi=min_midi, bpm_hint=bpm)
    effective_bpm = float(bpm) if bpm and bpm > 0 else float(trans["estimated_bpm"])
    scored = _build_score_artifacts(
        notes=trans["notes"],
        job_dir=job_dir,
        title=safe_title,
        bpm=effective_bpm,
        string_count=string_count,
    )
    tabbed: list[TabbedBassNote] = scored["tabbed_notes"]

    return {
        "status": "ok",
        "title": safe_title,
        "job_dir": str(job_dir),
        "duration_sec": acq["duration_sec"],
        "bpm": effective_bpm,
        "string_count": string_count,
        "engine": trans["engine"],
        "files": {
            "midi": str(scored["midi_path"]),
            "musicxml": str(scored["musicxml_path"]),
            "mscz": str(scored["mscz_path"]) if scored["mscz_path"] else None,
        },
        "note_count": len(tabbed),
        "chords": scored["chords"],
        "ascii_tab": scored["ascii_tab"],
    }


def _parse_notes_from_json(payload_text: str) -> tuple[str, list[TabbedBassNote]]:
    if len(payload_text.encode("utf-8")) > _MAX_PUSH_PAYLOAD_BYTES:
        raise SecurityValidationError("JSON payload exceeds 2 MB limit.")
    data = json.loads(payload_text)
    title = "Bass Score"
    raw_notes = data
    if isinstance(data, dict):
        title = _sanitize_title(str(data.get("title", "Bass Score")))
        raw_notes = data.get("notes", [])
    if not isinstance(raw_notes, list) or not raw_notes:
        raise SecurityValidationError("JSON payload must contain a non-empty list of notes.")

    notes: list[TabbedBassNote] = []
    for item in raw_notes:
        if not isinstance(item, dict):
            raise SecurityValidationError("Each note entry must be a JSON object.")
        midi = int(item["pitch_midi"]) if "pitch_midi" in item else int(item["midi"])
        if not (0 <= midi <= 127):
            raise SecurityValidationError("Note MIDI pitch must be in [0, 127].")
        string_num = int(item.get("string", item.get("string_num", 4)))
        fret = int(item.get("fret", 0))
        if not (1 <= string_num <= 5) or not (0 <= fret <= 24):
            raise SecurityValidationError("String must be 1..5 and fret must be 0..24.")
        dur = float(item.get("duration_sec", 0.5))
        onset = float(item.get("onset_sec", 0.0))
        vel = int(item.get("velocity", 96))
        notes.append(
            TabbedBassNote(
                pitch_midi=midi,
                note_name=str(item.get("note_name") or midi_to_note_name(midi)),
                onset_sec=max(0.0, onset),
                duration_sec=max(0.05, dur),
                velocity=max(1, min(127, vel)),
                string_num=string_num,
                fret=fret,
            )
        )
    return title, notes


def _parse_notes_from_musicxml(path_str: str) -> tuple[str, list[TabbedBassNote]]:
    env_dirs = parse_allowed_dirs(None)
    allowed_dirs = [_get_output_root(), *env_dirs] if env_dirs else None
    resolved = validate_input_file_path(
        file_path=path_str,
        allowed_extensions=frozenset({".musicxml", ".xml"}),
        max_bytes=_MAX_PUSH_PAYLOAD_BYTES,
        allowed_dirs=allowed_dirs,
        label="MusicXML",
    )

    xml_bytes = resolved.read_bytes()
    if _XML_DOCTYPE_RE.search(xml_bytes):
        raise SecurityValidationError("DOCTYPE and ENTITY declarations are rejected for XXE safety.")

    root = ET.fromstring(xml_bytes)
    title_el = root.find("./work/work-title")
    title = _sanitize_title(title_el.text if title_el is not None and title_el.text else "Bass Score")

    step_offsets = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    notes: list[TabbedBassNote] = []
    # Prefer staff 2 (TAB) if present, else read all pitched notes.
    seen_staff2 = False
    for note_el in root.findall(".//note"):
        staff_el = note_el.find("staff")
        if staff_el is not None and staff_el.text == "2":
            seen_staff2 = True
            break

    cursor_sec = 0.0
    for note_el in root.findall(".//note"):
        staff_el = note_el.find("staff")
        if seen_staff2 and (staff_el is None or staff_el.text != "2"):
            continue
        pitch_el = note_el.find("pitch")
        if pitch_el is None:
            continue
        step = (pitch_el.findtext("step") or "E").strip().upper()
        alter = int(pitch_el.findtext("alter") or "0")
        octave = int(pitch_el.findtext("octave") or "1")
        midi = (octave + 1) * 12 + step_offsets.get(step, 4) + alter
        s_num = int(note_el.findtext("./notations/technical/string") or "4")
        fret = int(note_el.findtext("./notations/technical/fret") or "0")
        notes.append(
            TabbedBassNote(
                pitch_midi=max(0, min(127, midi)),
                note_name=midi_to_note_name(max(0, min(127, midi))),
                onset_sec=round(cursor_sec, 4),
                duration_sec=0.5,
                velocity=96,
                string_num=max(1, min(5, s_num)),
                fret=max(0, min(24, fret)),
            )
        )
        cursor_sec += 0.5

    if not notes:
        raise ValueError("No pitched notes found in MusicXML file.")
    return title, notes


@mcp.tool()
async def musescore_live_push(musicxml_or_notes_json: str, port: int = 8765) -> dict[str, Any]:
    """Push transcribed bass notes or a MusicXML file to a local MuseScore 4 instance via 127.0.0.1."""
    stripped = (musicxml_or_notes_json or "").strip()
    if not stripped:
        raise SecurityValidationError("musicxml_or_notes_json must not be empty.")

    if stripped.startswith(("[", "{")):
        title, notes = _parse_notes_from_json(stripped)
    else:
        title, notes = _parse_notes_from_musicxml(stripped)

    bridge = MuseScoreBridge(host="127.0.0.1", port=port)
    result = await bridge.push_transcribed_notes(notes, title=title)
    return {"status": "ok", "title": title, **result}


@mcp.tool()
def check_environment_status() -> dict[str, Any]:
    """Report available external tools, ML engines, MuseScore CLI, and open-source attributions."""
    return {
        "status": "ok",
        "output_root": str(_get_output_root().resolve()),
        "dependencies": {
            "ffmpeg": shutil.which("ffmpeg"),
            "yt_dlp": importlib.util.find_spec("yt_dlp") is not None,
            "demucs": has_demucs(),
            "basic_pitch": has_basic_pitch(),
            "librosa": has_librosa(),
            "pretty_midi": importlib.util.find_spec("pretty_midi") is not None,
            "musescore_cli": find_musescore_binary(),
        },
        "attributions": [
            {"project": "spotify/basic-pitch", "license": "Apache-2.0"},
            {"project": "facebookresearch/demucs", "license": "MIT"},
            {"project": "ghchen99/mcp-musescore", "license": "MIT"},
            {"project": "sandraschi/stems-mcp", "license": "MIT"},
            {"project": "craffel/pretty-midi", "license": "MIT"},
            {"project": "yt-dlp/yt-dlp", "license": "The Unlicense"},
            {"project": "modelcontextprotocol/python-sdk", "license": "MIT"},
        ],
    }
