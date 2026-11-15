"""Audio acquisition from validated YouTube URLs or local audio files."""

from __future__ import annotations

import json
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from bass_mcp.metadata import parse_song_artist
from bass_mcp.security import (
    SecurityValidationError,
    SubprocessExecutionError,
    is_uri_input,
    run_safe_command,
    validate_input_audio_path,
    validate_youtube_url,
)

TARGET_SAMPLE_RATE = 44100


def _read_wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate) if rate > 0 else 0.0


def _normalize_wav_python(src_wav: Path, dst_wav: Path) -> None:
    """Normalize an input PCM WAV to 44.1kHz mono 16-bit PCM without ffmpeg."""
    sr, data = wavfile.read(str(src_wav))
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        audio = data.astype(np.float32) / max(max_val, 1.0)
    else:
        audio = np.clip(data.astype(np.float32), -1.0, 1.0)

    if sr != TARGET_SAMPLE_RATE and sr > 0:
        from math import gcd

        g = gcd(TARGET_SAMPLE_RATE, int(sr))
        audio = resample_poly(audio, TARGET_SAMPLE_RATE // g, int(sr) // g).astype(np.float32)

    pcm16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(str(dst_wav), TARGET_SAMPLE_RATE, pcm16)


def _convert_to_wav_ffmpeg(src_path: Path, dst_wav: Path) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise SubprocessExecutionError("ffmpeg is required on PATH to decode non-WAV audio files.")
    run_safe_command(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-i",
            str(src_path),
            "-vn",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(dst_wav),
        ],
        timeout=300,
    )


def _read_ytdlp_song_metadata(job_dir: Path, fallback_id: str) -> dict[str, str]:
    info_files = sorted(job_dir.glob("yt_download*.info.json"))
    for info_path in info_files:
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
                raw_title=str(payload.get("title") or fallback_id),
                uploader=str(payload.get("uploader") or payload.get("channel") or ""),
                track=str(payload.get("track") or ""),
                artist=str(payload.get("artist") or ""),
            )
    return parse_song_artist(raw_title=fallback_id)


def _download_youtube_audio(canonical_url: str, job_dir: Path, max_duration_sec: int) -> tuple[Path, dict[str, str]]:
    """Download audio from a canonicalized YouTube URL using yt-dlp in an isolated subprocess."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise SubprocessExecutionError("ffmpeg is required on PATH for yt-dlp audio extraction.")

    output_template = str(job_dir / "yt_download.%(ext)s")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--write-info-json",
        "--max-filesize",
        "100M",
        "--match-filter",
        f"duration <= {int(max_duration_sec)}",
        "--ffmpeg-location",
        ffmpeg_bin,
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        output_template,
        "--",
        canonical_url,
    ]
    run_safe_command(cmd, timeout=300, cwd=job_dir)

    candidates = sorted(job_dir.glob("yt_download*.wav"))
    if not candidates:
        raise SubprocessExecutionError("yt-dlp completed but no WAV audio file was produced.")
    downloaded_wav = candidates[0]
    normalized_wav = job_dir / "source_audio.wav"
    _convert_to_wav_ffmpeg(downloaded_wav, normalized_wav)
    song_meta = _read_ytdlp_song_metadata(job_dir, fallback_id=canonical_url)
    return normalized_wav, song_meta


def acquire_audio(
    source: str,
    job_dir: Path,
    max_duration_sec: int = 900,
    allowed_dirs: list[Path] | None = None,
) -> dict:
    """Acquire and normalize audio to 44.1kHz mono WAV from a YouTube URL or local file."""
    if max_duration_sec <= 0:
        raise SecurityValidationError("max_duration_sec must be positive.")

    out_wav = job_dir / "source_audio.wav"
    cleaned = source.strip() if isinstance(source, str) else ""

    if is_uri_input(cleaned) or cleaned.startswith("-"):
        video_id, canonical_url = validate_youtube_url(cleaned)
        wav_path, song_meta = _download_youtube_audio(canonical_url, job_dir, max_duration_sec)
        source_type = "youtube"
        source_ref = canonical_url
        meta_id = video_id
    else:
        local_path = validate_input_audio_path(cleaned, allowed_dirs=allowed_dirs)
        if shutil.which("ffmpeg"):
            _convert_to_wav_ffmpeg(local_path, out_wav)
        elif local_path.suffix.lower() == ".wav":
            _normalize_wav_python(local_path, out_wav)
        else:
            raise SubprocessExecutionError(
                f"ffmpeg is required to convert '{local_path.suffix}' files to WAV."
            )
        wav_path = out_wav
        source_type = "local_file"
        source_ref = str(local_path)
        meta_id = local_path.stem
        song_meta = parse_song_artist(raw_title=local_path.stem)

    duration_sec = _read_wav_duration_seconds(wav_path)
    if duration_sec <= 0:
        raise ValueError("Acquired audio file has zero duration.")
    if duration_sec > max_duration_sec:
        raise ValueError(
            f"Audio duration ({duration_sec:.1f}s) exceeds maximum allowed ({max_duration_sec}s)."
        )

    return {
        "source_type": source_type,
        "source_ref": source_ref,
        "id": meta_id,
        "wav_path": wav_path,
        "sample_rate": TARGET_SAMPLE_RATE,
        "duration_sec": round(duration_sec, 3),
        "song_metadata": song_meta,
    }
