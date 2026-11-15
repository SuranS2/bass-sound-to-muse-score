"""Security tests covering SSRF, URL injection, path traversal, devices, and env scrubbing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from bass_mcp.security import (
    SecurityValidationError,
    build_safe_env,
    format_job_dir_name,
    prepare_job_output_dir,
    rename_job_output_dir,
    run_safe_command,
    validate_input_audio_path,
    validate_youtube_url,
)


@pytest.mark.parametrize(
    ("raw_url", "expected_id"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("https://music.youtube.com/watch?v=abcDEF_1234&list=PL123", "abcDEF_1234"),
        ("https://m.youtube.com/shorts/Z9y_X1aB2c3", "Z9y_X1aB2c3"),
    ],
)
def test_validate_youtube_url_accepts_canonical_urls(raw_url: str, expected_id: str) -> None:
    vid, canonical = validate_youtube_url(raw_url)
    assert vid == expected_id
    assert canonical == f"https://www.youtube.com/watch?v={expected_id}"


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "http://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "file:///etc/passwd",
        "https://127.0.0.1/watch?v=dQw4w9WgXcQ",
        "https://169.254.169.254/latest/meta-data/",
        "https://evil-youtube.com/watch?v=dQw4w9WgXcQ",
        "https://user:pass@www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com:8443/watch?v=dQw4w9WgXcQ",
        "--exec=calc.exe",
        "https://www.youtube.com/playlist?list=PL123456",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ\n--postprocessor-args",
    ],
)
def test_validate_youtube_url_rejects_untrusted_inputs(bad_url: str) -> None:
    with pytest.raises(SecurityValidationError):
        validate_youtube_url(bad_url)


@pytest.mark.parametrize("device_path", ["CON", "CON.wav", "NUL.mp3", "COM1.wav", "LPT9.flac", "dir/AUX.wav"])
def test_validate_input_audio_path_rejects_windows_devices(device_path: str) -> None:
    with pytest.raises(SecurityValidationError):
        validate_input_audio_path(device_path)


def test_validate_input_audio_path_rejects_traversal_and_bad_extensions(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()

    valid_wav = allowed / "bass.wav"
    valid_wav.write_bytes(b"RIFF" + b"\x00" * 40)
    assert validate_input_audio_path(str(valid_wav), allowed_dirs=[allowed]) == valid_wav.resolve()

    secret_wav = outside / "secret.wav"
    secret_wav.write_bytes(b"RIFF" + b"\x00" * 40)
    traversal = str(allowed / ".." / "outside" / "secret.wav")
    with pytest.raises(SecurityValidationError):
        validate_input_audio_path(traversal, allowed_dirs=[allowed])

    bad_ext = allowed / "script.exe"
    bad_ext.write_bytes(b"MZ")
    with pytest.raises(SecurityValidationError):
        validate_input_audio_path(str(bad_ext), allowed_dirs=[allowed])


def test_validate_input_audio_path_rejects_symlinks_when_supported(tmp_path: Path) -> None:
    target = tmp_path / "real.wav"
    target.write_bytes(b"RIFF" + b"\x00" * 40)
    link = tmp_path / "link.wav"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation requires elevated privileges on this Windows host.")
    with pytest.raises(SecurityValidationError):
        validate_input_audio_path(str(link))


def test_prepare_job_output_dir_blocks_escape(tmp_path: Path) -> None:
    with pytest.raises(SecurityValidationError):
        prepare_job_output_dir(tmp_path, job_id="../escape")
    with pytest.raises(SecurityValidationError):
        prepare_job_output_dir(tmp_path, job_id="CON")

    job_dir = prepare_job_output_dir(tmp_path, job_id="song_01")
    assert job_dir.is_relative_to(tmp_path.resolve())
    assert job_dir.is_dir()


def test_subprocess_env_scrubs_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-super-secret")
    env = build_safe_env()
    assert "OPENAI_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env

    proc = run_safe_command(
        [sys.executable, "-c", "import os; print(os.environ.get('OPENAI_API_KEY', 'MISSING'))"]
    )
    assert proc.stdout.strip() == "MISSING"


def test_format_and_rename_job_output_dir_sanitizes_and_preserves_root(tmp_path: Path) -> None:
    name = format_job_dir_name(
        song_title="Marry Me? ../evil",
        artist="ELLEGARDEN / Band",
        timestamp="20261115_230000",
        short_id="ab12cd34",
    )
    assert name == "Marry_Me_evil-ELLEGARDEN_Band-20261115_230000-ab12cd34"

    raw_dir = prepare_job_output_dir(tmp_path)
    renamed = rename_job_output_dir(raw_dir, song_title="비밀곡", artist="가수")
    assert renamed.parent == tmp_path.resolve()
    assert renamed.is_dir()
    assert renamed.name.startswith("비밀곡-가수-")
