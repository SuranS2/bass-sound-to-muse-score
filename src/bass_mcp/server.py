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

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # pragma: no cover - mcp 1.x fallback
    from mcp.server.fastmcp import FastMCP

from bass_mcp.downloader import acquire_audio
from bass_mcp.metadata import lookup_original_song_bpm, lookup_song_lyrics, parse_song_artist
from bass_mcp.musescore_bridge import MuseScoreBridge
from bass_mcp.score_builder import (
    TabbedBassNote,
    assign_bass_tab_positions,
    compute_onset_shift_sec,
    convert_with_musescore_cli,
    find_musescore_binary,
    infer_measure_chords,
    normalize_note_onsets,
    parse_time_signature,
    render_ascii_bass_tab,
    resolve_time_signature,
    write_midi_file,
    write_musicxml_score,
)
from bass_mcp.security import (
    SecurityValidationError,
    parse_allowed_dirs,
    is_uri_input,
    prepare_job_output_dir,
    rename_job_output_dir,
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

_SERVER_INSTRUCTIONS = (
    "For YouTube/audio bass transcription, when the user has not explicitly specified a BPM, "
    "call transcribe_bass_pipeline with bpm=None first. It returns song_metadata (song_title, artist), "
    "audio_estimated_bpm, web_bpm_lookup, and a ready bass_wav stem with status='needs_bpm_confirmation'. "
    "If the track has an original song and web_bpm_lookup.bpm is null, also use your own web search "
    "(e.g. web_bpm_lookup.web_search_query) to check the original track's BPM. "
    "Always show the user the detected '곡 명 - 가수' and BPM, ask them to confirm or enter a corrected BPM, "
    "and then call audio_to_bass_score(audio_path=files['bass_wav'], title=<confirmed title>, bpm=<confirmed bpm>) "
    "to generate the final MusicXML, MSCZ, MIDI, and TAB without re-running stem separation."
)

mcp = FastMCP("bass-sound-to-musescore", instructions=_SERVER_INSTRUCTIONS)

_XML_DOCTYPE_RE = re.compile(rb"<!(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_MAX_PUSH_PAYLOAD_BYTES = 2 * 1024 * 1024


def _get_output_root() -> Path:
    configured = os.environ.get("BASS_MCP_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path.cwd() / "output"


def _effective_allowed_dirs() -> list[Path] | None:
    """When BASS_MCP_ALLOWED_DIRS is set, also permit files generated inside output_root."""
    env_dirs = parse_allowed_dirs(None)
    return [_get_output_root(), *env_dirs] if env_dirs else None


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
    lyrics: list[Any] | None = None,
    time_signature: str | None = None,
) -> dict[str, Any]:
    beats, beat_type = resolve_time_signature(
        notes,
        bpm=bpm,
        time_signature=time_signature if time_signature is not None else "auto",
    )
    shift_sec = compute_onset_shift_sec(
        notes, bpm=bpm, beats_per_measure=beats, preserve_intro_bars=True, beat_type=beat_type
    )
    aligned_notes = normalize_note_onsets(
        notes, bpm=bpm, beats_per_measure=beats, preserve_intro_bars=True, beat_type=beat_type
    )
    tabbed = assign_bass_tab_positions(aligned_notes, string_count=string_count)
    chords = infer_measure_chords(tabbed, bpm=bpm, beats_per_measure=beats, beat_type=beat_type)

    midi_path = write_midi_file(tabbed, job_dir / "bass_line.mid", bpm=bpm)
    musicxml_path = write_musicxml_score(
        tabbed,
        chords,
        job_dir / "bass_score.musicxml",
        title=title,
        bpm=bpm,
        string_count=string_count,
        beats_per_measure=beats,
        lyrics=lyrics,
        lyric_time_shift_sec=shift_sec,
        beat_type=beat_type,
    )
    ascii_tab = render_ascii_bass_tab(
        tabbed, chords, bpm=bpm, string_count=string_count, beats_per_measure=beats, beat_type=beat_type
    )
    mscz_path = convert_with_musescore_cli(musicxml_path, output_format="mscz")

    return {
        "tabbed_notes": tabbed,
        "chords": [{"measure": c.measure_num, "symbol": c.symbol} for c in chords],
        "time_signature": f"{beats}/{beat_type}",
        "midi_path": midi_path,
        "musicxml_path": musicxml_path,
        "mscz_path": mscz_path,
        "ascii_tab": ascii_tab,
    }


def _resolve_song_metadata(acq: dict[str, Any], requested_title: str) -> tuple[dict[str, str], str]:
    meta = dict(acq.get("song_metadata") or {})
    if not meta:
        meta = parse_song_artist(raw_title=str(acq.get("id") or requested_title))
    if requested_title.strip() and requested_title.strip() != "Bass Score":
        effective = _sanitize_title(requested_title)
        parsed_req = parse_song_artist(raw_title=effective)
        if not meta.get("song_title") or meta.get("song_title") == str(acq.get("id")):
            meta["song_title"] = parsed_req.get("song_title") or effective
        if not meta.get("artist") and parsed_req.get("artist"):
            meta["artist"] = parsed_req["artist"]
    else:
        effective = _sanitize_title(meta.get("display_name") or requested_title)
    return meta, effective


def _find_sibling_yt_metadata(audio_path: Path) -> dict[str, str] | None:
    """Inspect parent folders up to 4 levels for yt_download*.info.json metadata."""
    cur = audio_path.resolve().parent
    for _ in range(4):
        for info_path in sorted(cur.glob("yt_download*.info.json")):
            if info_path.is_symlink() or not info_path.is_file():
                continue
            if info_path.stat().st_size > 5 * 1024 * 1024:
                continue
            try:
                payload = json.loads(info_path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                return parse_song_artist(
                    raw_title=str(payload.get("title") or ""),
                    uploader=str(payload.get("uploader") or payload.get("channel") or ""),
                    track=str(payload.get("track") or ""),
                    artist=str(payload.get("artist") or ""),
                )
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _resolve_lyrics_and_naming(
    song_title: str,
    artist: str,
    duration_sec: float | None,
) -> tuple[str, str, list[Any]]:
    """Fetch lyrics via LRCLIB and return canonical (song_title, artist, lyrics_lines)."""
    clean_title = song_title.strip()
    clean_artist = artist.strip()
    if not clean_title or clean_title == "Bass Score":
        return clean_title, clean_artist, []
    lyrics_res = lookup_song_lyrics(
        song_title=clean_title,
        artist=clean_artist,
        duration_sec=duration_sec,
    )
    if not lyrics_res.get("found"):
        return clean_title, clean_artist, []
    canon_title = str(lyrics_res.get("canonical_title") or clean_title)
    canon_artist = str(lyrics_res.get("canonical_artist") or clean_artist)
    lines = lyrics_res.get("synced_lines") or lyrics_res.get("plain_lines") or []
    return canon_title, canon_artist, lines


def _rebase_job_dir(
    job_dir: Path,
    song_title: str,
    artist: str,
    *path_refs: tuple[dict[str, Any], str],
) -> Path:
    """Rename job_dir using song/artist metadata and update dependent file paths."""
    if not song_title and not artist:
        return job_dir
    new_dir = rename_job_output_dir(job_dir, song_title=song_title, artist=artist)
    if new_dir != job_dir:
        for container, key in path_refs:
            container[key] = new_dir / Path(container[key]).relative_to(job_dir)
    return new_dir


@mcp.tool()
def lookup_song_bpm(song_title: str, artist: str = "") -> dict[str, Any]:
    """Look up canonical song metadata and original BPM from allowlisted public music APIs."""
    return lookup_original_song_bpm(song_title=song_title, artist=artist)


@mcp.tool()
async def transcribe_bass_pipeline(
    source: str,
    title: str = "Bass Score",
    bpm: float | None = None,
    separation_method: str = "auto",
    string_count: int = 4,
    send_to_musescore: bool = False,
    auto_confirm_bpm: bool = False,
    time_signature: str | None = None,
) -> dict[str, Any]:
    """Convert a YouTube URL or local audio file into Bass MusicXML, 4/5-string TAB, chords, and MIDI.

    When bpm is omitted and auto_confirm_bpm is False, extracts song metadata, isolates the bass stem,
    checks public BPM lookup, and returns status='needs_bpm_confirmation' so the user can confirm or
    correct 'song_title - artist', BPM, and time_signature ('4/4', '3/4', '6/8', or 'auto') before
    generating final score files with audio_to_bass_score.
    """
    parse_time_signature(time_signature)
    min_midi = _min_midi_for_strings(string_count)
    job_dir = prepare_job_output_dir(_get_output_root())

    acq = acquire_audio(source=source, job_dir=job_dir)
    song_meta, safe_title = _resolve_song_metadata(acq, title)
    meta_song = song_meta.get("song_title") or (safe_title if safe_title != "Bass Score" else "")
    meta_artist = song_meta.get("artist") or ""
    job_dir = _rebase_job_dir(job_dir, meta_song, meta_artist, (acq, "wav_path"))

    sep = separate_bass_stem(acq["wav_path"], job_dir=job_dir, method=separation_method)
    trans = transcribe_bass_notes(sep["bass_wav"], min_midi=min_midi, bpm_hint=bpm)

    if (bpm is None or bpm <= 0) and not auto_confirm_bpm:
        web_lookup = lookup_original_song_bpm(
            song_title=song_meta.get("song_title", ""),
            artist=song_meta.get("artist", ""),
        )
        canonical_artist = web_lookup.get("canonical_artist") or song_meta.get("artist") or ""
        canonical_title = web_lookup.get("canonical_title") or song_meta.get("song_title") or safe_title
        display_name = (
            f"{canonical_artist} - {canonical_title}"
            if canonical_artist and canonical_title
            else safe_title
        )
        job_dir = _rebase_job_dir(
            job_dir, canonical_title, canonical_artist, (acq, "wav_path"), (sep, "bass_wav")
        )
        suggested_bpm = web_lookup["bpm"] if web_lookup.get("bpm") else trans["estimated_bpm"]
        rec_beats, rec_beat_type = resolve_time_signature(
            trans["notes"],
            bpm=float(suggested_bpm),
            time_signature=time_signature if time_signature is not None else "auto",
        )
        rec_time_sig = f"{rec_beats}/{rec_beat_type}"
        return {
            "status": "needs_bpm_confirmation",
            "requires_user_confirmation": True,
            "title": display_name,
            "job_dir": str(job_dir),
            "source_type": acq["source_type"],
            "source_ref": acq["source_ref"],
            "duration_sec": acq["duration_sec"],
            "string_count": string_count,
            "song_metadata": {
                **song_meta,
                "canonical_artist": canonical_artist,
                "canonical_title": canonical_title,
                "display_name": display_name,
            },
            "audio_estimated_bpm": trans["estimated_bpm"],
            "web_bpm_lookup": web_lookup,
            "recommended_bpm": suggested_bpm,
            "recommended_time_signature": rec_time_sig,
            "engines": {
                "separator": sep["engine"],
                "transcriber": trans["engine"],
            },
            "note_count_preview": len(trans["notes"]),
            "files": {
                "source_wav": str(acq["wav_path"]),
                "bass_wav": str(sep["bass_wav"]),
            },
            "confirmation_prompt_ko": (
                f"감지된 곡 명-가수: {display_name}\n"
                f"추정 BPM: {suggested_bpm} BPM (오디오 자동 추정: {trans['estimated_bpm']} BPM)\n"
                f"추천 박자표: {rec_time_sig} (선택하지 않으면 AI 자동 추정 적용)\n"
                "원본 곡 명-가수, BPM, 박자표가 맞는지 확인해 주세요. 수정할 항목이 있다면 입력해 주세요."
            ),
            "next_step_instructions": (
                "If this track has an original release and web_bpm_lookup.bpm is null, run a web search "
                "using web_bpm_lookup.web_search_query to check the original BPM. Then present the song "
                "name - artist, BPM, and recommended_time_signature to the user, ask for confirmation or "
                "a corrected BPM / time_signature, and call audio_to_bass_score(audio_path=files['bass_wav'], "
                "title=<confirmed_title>, bpm=<confirmed_bpm>, string_count=..., time_signature=<chosen or None>) "
                "to finalize the score."
            ),
        }

    canon_song, canon_artist, lyric_lines = _resolve_lyrics_and_naming(
        song_title=meta_song,
        artist=meta_artist,
        duration_sec=acq["duration_sec"],
    )
    job_dir = _rebase_job_dir(
        job_dir, canon_song, canon_artist, (acq, "wav_path"), (sep, "bass_wav")
    )

    effective_bpm = float(bpm) if bpm and bpm > 0 else float(trans["estimated_bpm"])
    scored = _build_score_artifacts(
        notes=trans["notes"],
        job_dir=job_dir,
        title=safe_title,
        bpm=effective_bpm,
        string_count=string_count,
        lyrics=lyric_lines,
        time_signature=time_signature,
    )

    push_result: dict[str, Any] | None = None
    if send_to_musescore:
        bridge = MuseScoreBridge(host="127.0.0.1", port=8765)
        push_result = await bridge.push_transcribed_notes(scored["tabbed_notes"], title=safe_title)

    tabbed: list[TabbedBassNote] = scored["tabbed_notes"]
    return {
        "status": "ok",
        "title": safe_title,
        "song_metadata": song_meta,
        "job_dir": str(job_dir),
        "source_type": acq["source_type"],
        "source_ref": acq["source_ref"],
        "duration_sec": acq["duration_sec"],
        "bpm": effective_bpm,
        "time_signature": scored["time_signature"],
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
        "lyrics_count": len(lyric_lines),
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
    meta = dict(acq.get("song_metadata") or {})
    job_dir = _rebase_job_dir(
        job_dir, meta.get("song_title") or "", meta.get("artist") or "", (acq, "wav_path")
    )
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
    time_signature: str | None = None,
) -> dict[str, Any]:
    """Convert an already-isolated local bass audio file into MusicXML, TAB, chords, and MIDI."""
    parse_time_signature(time_signature)
    if is_uri_input(audio_path):
        raise SecurityValidationError("audio_to_bass_score accepts local file paths only, not URLs.")
    allowed_dirs = _effective_allowed_dirs()
    resolved_audio = validate_input_audio_path(audio_path, allowed_dirs=allowed_dirs)

    safe_title = _sanitize_title(title)
    min_midi = _min_midi_for_strings(string_count)
    sibling_meta = _find_sibling_yt_metadata(resolved_audio) or {}
    parsed_title = parse_song_artist(raw_title=safe_title) if safe_title != "Bass Score" else {}
    init_song = (parsed_title.get("song_title") or sibling_meta.get("song_title") or "").strip()
    init_artist = (parsed_title.get("artist") or sibling_meta.get("artist") or "").strip()
    if safe_title == "Bass Score" and sibling_meta.get("display_name"):
        safe_title = _sanitize_title(sibling_meta["display_name"])

    job_dir = prepare_job_output_dir(_get_output_root(), song_title=init_song, artist=init_artist)

    acq = acquire_audio(source=audio_path, job_dir=job_dir, allowed_dirs=allowed_dirs)
    canon_song, canon_artist, lyric_lines = _resolve_lyrics_and_naming(
        song_title=init_song or sibling_meta.get("song_title") or "",
        artist=init_artist or sibling_meta.get("artist") or "",
        duration_sec=acq["duration_sec"],
    )
    job_dir = _rebase_job_dir(job_dir, canon_song, canon_artist, (acq, "wav_path"))

    trans = transcribe_bass_notes(acq["wav_path"], min_midi=min_midi, bpm_hint=bpm)
    effective_bpm = float(bpm) if bpm and bpm > 0 else float(trans["estimated_bpm"])
    scored = _build_score_artifacts(
        notes=trans["notes"],
        job_dir=job_dir,
        title=safe_title,
        bpm=effective_bpm,
        string_count=string_count,
        lyrics=lyric_lines,
        time_signature=time_signature,
    )
    tabbed: list[TabbedBassNote] = scored["tabbed_notes"]

    return {
        "status": "ok",
        "title": safe_title,
        "job_dir": str(job_dir),
        "duration_sec": acq["duration_sec"],
        "bpm": effective_bpm,
        "time_signature": scored["time_signature"],
        "string_count": string_count,
        "engine": trans["engine"],
        "files": {
            "midi": str(scored["midi_path"]),
            "musicxml": str(scored["musicxml_path"]),
            "mscz": str(scored["mscz_path"]) if scored["mscz_path"] else None,
        },
        "note_count": len(tabbed),
        "lyrics_count": len(lyric_lines),
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
    resolved = validate_input_file_path(
        file_path=path_str,
        allowed_extensions=frozenset({".musicxml", ".xml"}),
        max_bytes=_MAX_PUSH_PAYLOAD_BYTES,
        allowed_dirs=_effective_allowed_dirs(),
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
