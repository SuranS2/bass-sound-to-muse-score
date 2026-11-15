"""Security guards for URL validation, local path checks, and subprocess execution."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ALLOWED_YOUTUBE_HOSTS = frozenset(
    {
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)
ALLOWED_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})
MAX_AUDIO_FILE_BYTES = 200 * 1024 * 1024
MAX_URL_LENGTH = 2048

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_JOB_ID_RE = re.compile(r"^[0-9A-Za-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af_-]{1,120}$")
_SAFE_SLUG_STRIP_RE = re.compile(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+")
_JOB_SUFFIX_RE = re.compile(r"^(?:.+[-_])?(\d{8}_\d{6})[-_]([0-9a-fA-F]{8})$")
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

_SAFE_ENV_KEYS = (
    "PATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "HOME",
    "APPDATA",
    "LOCALAPPDATA",
    "PATHEXT",
    "COMSPEC",
)


class SecurityValidationError(ValueError):
    """Raised when an untrusted input violates a security boundary."""


class SubprocessExecutionError(RuntimeError):
    """Raised when a sandboxed external command fails."""


def is_uri_input(value: str) -> bool:
    """Return True if value looks like a URI with an explicit scheme."""
    return bool(_URI_SCHEME_RE.match(value.strip()))


def _check_no_control_chars(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise SecurityValidationError(f"{label} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise SecurityValidationError(f"{label} must not be empty.")
    if _CONTROL_CHARS_RE.search(cleaned):
        raise SecurityValidationError(f"{label} contains disallowed control characters.")
    return cleaned


def _check_not_windows_device_name(name: str, label: str) -> None:
    stem = name.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        raise SecurityValidationError(f"{label} uses reserved device name: {name}")


def validate_youtube_url(url: str) -> tuple[str, str]:
    """Validate and canonicalize a YouTube URL to prevent SSRF and flag injection.

    Returns ``(video_id, canonical_watch_url)``.
    """
    cleaned = _check_no_control_chars(url, "YouTube URL")
    if len(cleaned) > MAX_URL_LENGTH:
        raise SecurityValidationError("YouTube URL exceeds maximum allowed length.")
    if cleaned.startswith("-"):
        raise SecurityValidationError("YouTube URL must not start with '-'.")

    parsed = urlparse(cleaned)
    if parsed.scheme.lower() != "https":
        raise SecurityValidationError("Only https:// YouTube URLs are permitted.")
    if parsed.username is not None or parsed.password is not None:
        raise SecurityValidationError("Credentials in URLs are not permitted.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise SecurityValidationError("Invalid port in URL.") from exc
    if port not in (None, 443):
        raise SecurityValidationError("Non-standard ports are not permitted.")

    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_YOUTUBE_HOSTS:
        raise SecurityValidationError(f"Host '{host}' is not an allowed YouTube domain.")

    video_id: str | None = None
    if host == "youtu.be":
        segment = parsed.path.strip("/")
        if "/" not in segment and _VIDEO_ID_RE.fullmatch(segment):
            video_id = segment
    elif parsed.path == "/watch":
        v_values = parse_qs(parsed.query).get("v", [])
        if len(v_values) == 1 and _VIDEO_ID_RE.fullmatch(v_values[0]):
            video_id = v_values[0]
    else:
        for prefix in ("/shorts/", "/live/", "/embed/"):
            if parsed.path.startswith(prefix):
                candidate = parsed.path[len(prefix) :].strip("/")
                if "/" not in candidate and _VIDEO_ID_RE.fullmatch(candidate):
                    video_id = candidate
                break

    if not video_id:
        raise SecurityValidationError("Could not extract a valid 11-character YouTube video ID.")

    return video_id, f"https://www.youtube.com/watch?v={video_id}"


def parse_allowed_dirs(allowed_dirs: list[Path] | None = None) -> list[Path]:
    if allowed_dirs is not None:
        return [Path(d).resolve() for d in allowed_dirs]
    env_dirs = os.environ.get("BASS_MCP_ALLOWED_DIRS", "").strip()
    if not env_dirs:
        return []
    return [Path(p).resolve() for p in env_dirs.split(os.pathsep) if p.strip()]


def validate_input_file_path(
    file_path: str,
    allowed_extensions: frozenset[str],
    max_bytes: int,
    allowed_dirs: list[Path] | None = None,
    label: str = "File",
) -> Path:
    """Validate a local input path against traversal, symlinks, devices, and size limits."""
    cleaned = _check_no_control_chars(file_path, f"{label} path")
    if is_uri_input(cleaned):
        raise SecurityValidationError("URI schemes are not accepted as local file paths.")
    if cleaned.startswith("\\\\") or cleaned.startswith("//"):
        raise SecurityValidationError("UNC network paths are not permitted.")

    raw = Path(cleaned)
    for part in raw.parts:
        _check_not_windows_device_name(part, f"{label} path component")

    if raw.is_symlink():
        raise SecurityValidationError(f"Symbolic links are not permitted as {label.lower()} input.")

    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise SecurityValidationError(f"{label} file does not exist or cannot be resolved: {cleaned}") from exc

    _check_not_windows_device_name(resolved.name, f"Resolved {label.lower()} filename")
    if resolved.is_symlink():
        raise SecurityValidationError(f"Resolved {label.lower()} path must not be a symlink.")

    st = resolved.stat()
    if not stat.S_ISREG(st.st_mode):
        raise SecurityValidationError(f"{label} path must point to a regular file.")
    if resolved.suffix.lower() not in allowed_extensions:
        raise SecurityValidationError(
            f"Unsupported extension '{resolved.suffix}'. Allowed: {sorted(allowed_extensions)}"
        )
    if st.st_size <= 0 or st.st_size > max_bytes:
        raise SecurityValidationError(
            f"{label} file size ({st.st_size} bytes) must be between 1 byte and {max_bytes} bytes."
        )

    effective_dirs = parse_allowed_dirs(allowed_dirs)
    if effective_dirs and not any(resolved.is_relative_to(d) for d in effective_dirs):
        raise SecurityValidationError(f"{label} file lies outside configured allowed directories.")

    return resolved


def validate_input_audio_path(
    file_path: str,
    allowed_dirs: list[Path] | None = None,
) -> Path:
    """Validate a local audio path against traversal, symlinks, devices, and size limits."""
    return validate_input_file_path(
        file_path=file_path,
        allowed_extensions=ALLOWED_AUDIO_EXTENSIONS,
        max_bytes=MAX_AUDIO_FILE_BYTES,
        allowed_dirs=allowed_dirs,
        label="Audio",
    )


def _slugify_label(value: str, max_len: int) -> str:
    cleaned = _SAFE_SLUG_STRIP_RE.sub("_", str(value or "")).strip("_-")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("_-")
    return cleaned


def format_job_dir_name(
    song_title: str = "",
    artist: str = "",
    timestamp: str | None = None,
    short_id: str | None = None,
) -> str:
    """Build a filesystem-safe directory name as '<song>-<artist>-<YYYYMMDD_HHMMSS>-<id>'."""
    ts = timestamp if timestamp and re.fullmatch(r"\d{8}_\d{6}", timestamp) else time.strftime("%Y%m%d_%H%M%S")
    uid = short_id if short_id and re.fullmatch(r"[0-9a-fA-F]{8}", short_id) else uuid.uuid4().hex[:8]

    clean_title = _slugify_label(song_title, 36)
    clean_artist = _slugify_label(artist, 24)
    if clean_title and clean_artist and clean_artist.lower() != clean_title.lower():
        prefix = f"{clean_title}-{clean_artist}"
    elif clean_title:
        prefix = clean_title
    elif clean_artist:
        prefix = clean_artist
    else:
        prefix = "job"

    candidate = f"{prefix}-{ts}-{uid}"
    _check_not_windows_device_name(candidate, "job_id")
    if not _JOB_ID_RE.fullmatch(candidate):
        return f"job-{ts}-{uid}"
    return candidate


def prepare_job_output_dir(
    base_output_dir: Path,
    job_id: str | None = None,
    song_title: str = "",
    artist: str = "",
) -> Path:
    """Create and return an isolated job subdirectory strictly under base_output_dir."""
    base = Path(base_output_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    if job_id is None:
        if song_title.strip() or artist.strip():
            safe_id = format_job_dir_name(song_title=song_title, artist=artist)
        else:
            safe_id = f"job-{time.strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"
    else:
        safe_id = _check_no_control_chars(job_id, "job_id")
        if not _JOB_ID_RE.fullmatch(safe_id):
            raise SecurityValidationError("job_id contains disallowed characters or length.")
        _check_not_windows_device_name(safe_id, "job_id")

    job_dir = (base / safe_id).resolve()
    if job_dir == base or not job_dir.is_relative_to(base):
        raise SecurityValidationError("Resolved job directory escaped base output directory.")
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def rename_job_output_dir(
    job_dir: Path,
    song_title: str = "",
    artist: str = "",
) -> Path:
    """Safely rename an existing job subdirectory once song/artist metadata is known."""
    resolved = Path(job_dir).resolve()
    if not resolved.is_dir():
        return resolved
    base = resolved.parent.resolve()
    if resolved == base or not resolved.is_relative_to(base):
        raise SecurityValidationError("Resolved job directory escaped base output directory.")

    match = _JOB_SUFFIX_RE.fullmatch(resolved.name)
    ts = match.group(1) if match else None
    uid = match.group(2) if match else None
    new_name = format_job_dir_name(song_title=song_title, artist=artist, timestamp=ts, short_id=uid)
    target = (base / new_name).resolve()
    if target == base or not target.is_relative_to(base):
        raise SecurityValidationError("Renamed job directory escaped base output directory.")
    if target == resolved:
        return resolved
    if target.exists():
        new_name = format_job_dir_name(song_title=song_title, artist=artist, timestamp=ts, short_id=uuid.uuid4().hex[:8])
        target = (base / new_name).resolve()
    resolved.rename(target)
    return target


def build_safe_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a minimal allowlisted environment dictionary without inherited secrets."""
    env: dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if extra_env:
        env.update(extra_env)
    return env


def run_safe_command(
    args: list[str],
    timeout: int = 300,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Execute a subprocess safely without a shell and with scrubbed environment variables."""
    if not isinstance(args, list) or not args:
        raise SecurityValidationError("Command arguments must be a non-empty list.")
    clean_args: list[str] = []
    for arg in args:
        if not isinstance(arg, str) or not arg or "\x00" in arg:
            raise SecurityValidationError("Each command argument must be a non-empty string without null bytes.")
        clean_args.append(arg)

    proc = subprocess.run(
        clean_args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
        timeout=timeout,
        cwd=str(cwd) if cwd is not None else None,
        env=build_safe_env(),
    )
    if check and proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-800:]
        raise SubprocessExecutionError(
            f"Command {Path(clean_args[0]).name} failed with exit code {proc.returncode}: {stderr_tail}"
        )
    return proc
