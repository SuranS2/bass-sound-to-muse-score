"""Song metadata extraction and allowlisted public BPM lookup."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from bass_mcp.security import SecurityValidationError

_ALLOWED_API_HOSTS = frozenset({"itunes.apple.com", "api.deezer.com", "lrclib.net"})
_LRC_LINE_RE = re.compile(r"^((?:\[\d{1,3}:\d{2}(?:\.\d{1,3})?\])+)(.*)$")
_LRC_TIME_RE = re.compile(r"\[(\d{1,3}):(\d{2})(?:\.(\d{1,3}))?\]")


@dataclass(frozen=True)
class LyricLine:
    """Single timestamped lyric phrase aligned to song time in seconds."""

    onset_sec: float
    text: str
_MAX_HTTP_BYTES = 512 * 1024
_USER_AGENT = "bass-sound-to-muse-score/0.1 (local-mcp)"

_COVER_NOISE_RE = re.compile(
    r"(?:\|.*$|【[^】]*】|\[[^\]]*\]|\((?:bass|guitar|drum|vocal|instrumental|cover|tab|악보|커버)[^)]*\))",
    re.IGNORECASE,
)
_TRAILING_TAGS_RE = re.compile(
    r"\b(?:bass\s*(?:guitar)?\s*(?:cover|tutorial|tab|playthrough|lesson)|cover\s*ver\.?|베이스\s*커버|베이스\s*악보|베이스|커버|악보)\b",
    re.IGNORECASE,
)
_LEADING_NUM_RE = re.compile(r"^\s*(?:\[\d+\]|\d+[.)\s-]+)")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _clean_text(value: str, max_len: int = 160) -> str:
    cleaned = _CONTROL_RE.sub(" ", str(value or "")).strip()
    return re.sub(r"\s+", " ", cleaned)[:max_len]


def _strip_cover_noise(text: str) -> str:
    cleaned = _COVER_NOISE_RE.sub(" ", text)
    cleaned = _TRAILING_TAGS_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" -_,|/")


def parse_song_artist(
    raw_title: str,
    uploader: str = "",
    track: str = "",
    artist: str = "",
) -> dict[str, str]:
    """Extract canonical (artist, song_title) from YouTube metadata or audio filename."""
    clean_raw = _clean_text(raw_title, max_len=240)
    meta_track = _clean_text(track)
    meta_artist = _clean_text(artist)

    if meta_track and meta_artist:
        song_title = _strip_cover_noise(meta_track) or meta_track
        artist_name = _strip_cover_noise(meta_artist) or meta_artist
    else:
        stripped = _strip_cover_noise(clean_raw)
        artist_name = ""
        song_title = ""
        for sep in (" - ", " – "):
            if sep in stripped:
                left, right = stripped.split(sep, 1)
                left = _LEADING_NUM_RE.sub("", left).strip()
                if "," in left:
                    left = left.split(",")[-1].strip()
                artist_name = left.strip(" -_,")
                song_title = right.strip(" -_,")
                break
        if not song_title:
            song_title = _LEADING_NUM_RE.sub("", stripped).strip() or clean_raw

    display_name = f"{artist_name} - {song_title}" if artist_name and song_title else (song_title or clean_raw)
    search_terms = f"{artist_name} {song_title} BPM".strip()
    return {
        "raw_title": clean_raw,
        "artist": artist_name,
        "song_title": song_title,
        "uploader": _clean_text(uploader),
        "display_name": display_name,
        "web_search_query": search_terms,
    }


def _fetch_allowlisted_json(url: str, timeout_sec: float = 6.0) -> Any:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in _ALLOWED_API_HOSTS:
        raise SecurityValidationError(f"Disallowed metadata lookup URL host: {parsed.hostname}")

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        final_host = (urllib.parse.urlparse(resp.geturl()).hostname or "").lower()
        if final_host not in _ALLOWED_API_HOSTS:
            raise SecurityValidationError(f"Redirected to disallowed host: {final_host}")
        raw = resp.read(_MAX_HTTP_BYTES)
    return json.loads(raw.decode("utf-8", errors="replace"))


def lookup_original_song_bpm(
    song_title: str,
    artist: str = "",
    timeout_sec: float = 6.0,
) -> dict[str, Any]:
    """Check allowlisted music metadata APIs for original artist/title and BPM if indexed."""
    clean_title = _clean_text(song_title)
    clean_artist = _clean_text(artist)
    query = f"{clean_artist} {clean_title}".strip()
    web_query = f"{query} BPM".strip() if query else ""

    result: dict[str, Any] = {
        "query": query,
        "web_search_query": web_query,
        "found": False,
        "bpm": None,
        "source": None,
        "canonical_artist": clean_artist or None,
        "canonical_title": clean_title or None,
    }
    if not query:
        return result

    encoded = urllib.parse.quote_plus(query)
    try:
        itunes = _fetch_allowlisted_json(
            f"https://itunes.apple.com/search?term={encoded}&media=music&limit=3",
            timeout_sec=timeout_sec,
        )
        items = itunes.get("results") if isinstance(itunes, dict) else None
        if isinstance(items, list) and items:
            top = items[0]
            result["canonical_artist"] = top.get("artistName") or result["canonical_artist"]
            result["canonical_title"] = top.get("trackName") or result["canonical_title"]
    except (OSError, ValueError, json.JSONDecodeError, SecurityValidationError):
        pass

    try:
        deezer = _fetch_allowlisted_json(
            f"https://api.deezer.com/search?q={encoded}&limit=3",
            timeout_sec=timeout_sec,
        )
        tracks = deezer.get("data") if isinstance(deezer, dict) else None
        if isinstance(tracks, list):
            for track in tracks[:3]:
                tid = track.get("id")
                if tid is None:
                    continue
                detail = _fetch_allowlisted_json(
                    f"https://api.deezer.com/track/{int(tid)}",
                    timeout_sec=timeout_sec,
                )
                bpm_val = detail.get("bpm") if isinstance(detail, dict) else None
                if isinstance(bpm_val, (int, float)) and 35.0 <= float(bpm_val) <= 260.0:
                    result["found"] = True
                    result["bpm"] = round(float(bpm_val), 1)
                    result["source"] = "deezer:api"
                    if not result["canonical_title"] and detail.get("title"):
                        result["canonical_title"] = str(detail["title"])
                    artist_obj = detail.get("artist") if isinstance(detail, dict) else None
                    if not result["canonical_artist"] and isinstance(artist_obj, dict) and artist_obj.get("name"):
                        result["canonical_artist"] = str(artist_obj["name"])
                    break
    except (OSError, ValueError, TypeError, json.JSONDecodeError, SecurityValidationError):
        pass

    return result


def parse_lrc_lyrics(lrc_text: str) -> list[LyricLine]:
    """Parse LRC '[mm:ss.xx] text' lines into sorted LyricLine events."""
    lines: list[LyricLine] = []
    for raw_line in str(lrc_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _LRC_LINE_RE.match(line)
        if not match:
            continue
        tags_part, text_part = match.groups()
        clean_phrase = _clean_text(text_part, max_len=120)
        if not clean_phrase or clean_phrase.startswith("[") and clean_phrase.endswith("]"):
            continue
        for mm_s, ss_s, frac_s in _LRC_TIME_RE.findall(tags_part):
            sec = int(mm_s) * 60 + int(ss_s)
            if frac_s:
                sec += int(frac_s) / (10 ** len(frac_s))
            lines.append(LyricLine(onset_sec=round(float(sec), 3), text=clean_phrase))
    lines.sort(key=lambda item: item.onset_sec)
    return lines


def lookup_song_lyrics(
    song_title: str,
    artist: str = "",
    duration_sec: float | None = None,
    timeout_sec: float = 6.0,
) -> dict[str, Any]:
    """Fetch synced or plain lyrics from allowlisted lrclib.net when indexed."""
    clean_title = _clean_text(song_title)
    clean_artist = _clean_text(artist)
    query = f"{clean_artist} {clean_title}".strip()
    empty_res: dict[str, Any] = {
        "found": False,
        "synced_lines": [],
        "plain_lines": [],
        "canonical_artist": clean_artist or None,
        "canonical_title": clean_title or None,
        "source": None,
    }
    if not query:
        return empty_res

    encoded = urllib.parse.quote_plus(query)
    try:
        items = _fetch_allowlisted_json(
            f"https://lrclib.net/api/search?q={encoded}",
            timeout_sec=timeout_sec,
        )
    except (OSError, ValueError, json.JSONDecodeError, SecurityValidationError):
        return empty_res

    if not isinstance(items, list) or not items:
        return empty_res

    tokens = [
        tok.lower()
        for tok in re.split(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+", query)
        if len(tok) >= 2
    ]
    best_item: dict[str, Any] | None = None
    best_score = -float("inf")
    for item in items[:10]:
        if not isinstance(item, dict):
            continue
        synced = str(item.get("syncedLyrics") or "").strip()
        plain = str(item.get("plainLyrics") or "").strip()
        if not synced and not plain:
            continue
        haystack = f"{item.get('trackName', '')} {item.get('artistName', '')}".lower()
        score = 4.0 if synced else 1.0
        score += sum(1.5 for tok in tokens if tok in haystack)
        item_dur = item.get("duration")
        if duration_sec and isinstance(item_dur, (int, float)) and item_dur > 0:
            score -= min(4.0, abs(float(item_dur) - float(duration_sec)) / 10.0)
        if score > best_score:
            best_score = score
            best_item = item

    if not best_item:
        return empty_res

    synced_lines = parse_lrc_lyrics(str(best_item.get("syncedLyrics") or ""))
    plain_lines = [
        _clean_text(line, max_len=120)
        for line in str(best_item.get("plainLyrics") or "").splitlines()
        if _clean_text(line, max_len=120)
    ]
    return {
        "found": bool(synced_lines or plain_lines),
        "synced_lines": synced_lines,
        "plain_lines": plain_lines,
        "canonical_artist": _clean_text(str(best_item.get("artistName") or "")) or (clean_artist or None),
        "canonical_title": _clean_text(str(best_item.get("trackName") or "")) or (clean_title or None),
        "source": "lrclib:api",
    }
