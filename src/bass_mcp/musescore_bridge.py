"""Loopback-only WebSocket client for live MuseScore 4 score updates.

Adapted from ghchen99/mcp-musescore (MIT License, Copyright (c) 2025 ghchen99),
hardened to strictly permit loopback connections only (127.0.0.1).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import websockets

from bass_mcp.score_builder import TabbedBassNote
from bass_mcp.security import SecurityValidationError

ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})


def validate_musescore_endpoint(host: str, port: int) -> str:
    """Validate that the MuseScore bridge endpoint is strictly local loopback."""
    cleaned_host = (host or "").strip().lower()
    if cleaned_host not in ALLOWED_LOOPBACK_HOSTS:
        raise SecurityValidationError(
            f"Refusing non-loopback MuseScore host '{host}'. Only 127.0.0.1 / localhost is permitted."
        )
    if not isinstance(port, int) or not (1024 <= port <= 65535):
        raise SecurityValidationError("MuseScore WebSocket port must be an integer between 1024 and 65535.")
    canonical_host = "127.0.0.1"
    return f"ws://{canonical_host}:{port}"


class MuseScoreBridge:
    """Async WebSocket client talking to the local MuseScore 4 QML WebSocket plugin."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout: float = 5.0) -> None:
        self.url = validate_musescore_endpoint(host, port)
        self.timeout = timeout

    async def send_command(self, action: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a single JSON command to the local MuseScore WebSocket plugin."""
        payload = json.dumps({"action": action, "params": params or {}})
        async with websockets.connect(self.url, open_timeout=self.timeout, close_timeout=2.0) as ws:
            await ws.send(payload)
            raw_reply = await ws.recv()
        try:
            data = json.loads(raw_reply)
        except json.JSONDecodeError:
            return {"status": "ok", "raw": str(raw_reply)}
        return data if isinstance(data, dict) else {"status": "ok", "result": data}

    async def send_commands_batch(self, commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Send a sequence of JSON commands over a single local WebSocket session."""
        results: list[dict[str, Any]] = []
        async with websockets.connect(self.url, open_timeout=self.timeout, close_timeout=2.0) as ws:
            for cmd in commands:
                await ws.send(json.dumps(cmd))
                raw_reply = await ws.recv()
                try:
                    parsed = json.loads(raw_reply)
                except json.JSONDecodeError:
                    parsed = {"status": "ok", "raw": str(raw_reply)}
                results.append(parsed)
        return results

    async def push_transcribed_notes(
        self,
        notes: list[TabbedBassNote],
        title: str = "Bass Transcription",
    ) -> dict[str, Any]:
        """Push transcribed bass notes into the active MuseScore 4 score."""
        commands: list[dict[str, Any]] = [
            {"action": "set_title", "params": {"title": title}},
        ]
        for note in notes:
            commands.append(
                {
                    "action": "add_note",
                    "params": {
                        "pitch": int(note.pitch_midi),
                        "duration": round(float(note.duration_sec), 3),
                        "string": int(note.string_num),
                        "fret": int(note.fret),
                        "note_meta": asdict(note),
                    },
                }
            )
        replies = await self.send_commands_batch(commands)
        return {
            "pushed_notes": len(notes),
            "endpoint": self.url,
            "replies_count": len(replies),
        }
