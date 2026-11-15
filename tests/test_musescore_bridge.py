"""Tests for loopback-only MuseScore WebSocket bridge and XXE-safe live push."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import websockets

from bass_mcp.musescore_bridge import MuseScoreBridge, validate_musescore_endpoint
from bass_mcp.score_builder import TabbedBassNote
from bass_mcp.security import SecurityValidationError
from bass_mcp.server import musescore_live_push


@pytest.mark.parametrize(
    ("host", "port"),
    [
        ("0.0.0.0", 8765),
        ("192.168.1.50", 8765),
        ("example.com", 8765),
        ("127.0.0.1", 80),
        ("127.0.0.1", 70000),
    ],
)
def test_validate_musescore_endpoint_rejects_non_loopback_and_bad_ports(host: str, port: int) -> None:
    with pytest.raises(SecurityValidationError):
        validate_musescore_endpoint(host, port)


def test_validate_musescore_endpoint_normalizes_localhost() -> None:
    assert validate_musescore_endpoint("localhost", 8765) == "ws://127.0.0.1:8765"
    assert validate_musescore_endpoint("127.0.0.1", 8765) == "ws://127.0.0.1:8765"


@pytest.mark.asyncio
async def test_musescore_bridge_and_live_push_with_local_mock_ws() -> None:
    received: list[dict] = []

    async def handler(ws) -> None:
        async for msg in ws:
            payload = json.loads(msg)
            received.append(payload)
            await ws.send(json.dumps({"status": "ok", "action": payload.get("action")}))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]

        bridge = MuseScoreBridge(host="127.0.0.1", port=port)
        res = await bridge.push_transcribed_notes(
            [TabbedBassNote(28, "E1", 0.0, 0.5, 96, 4, 0)],
            title="Live Test",
        )
        assert res["pushed_notes"] == 1
        assert [cmd["action"] for cmd in received] == ["set_title", "add_note"]

        received.clear()
        json_payload = json.dumps(
            {
                "title": "JSON Push",
                "notes": [{"pitch_midi": 33, "string": 3, "fret": 0, "duration_sec": 0.5}],
            }
        )
        push_res = await musescore_live_push(json_payload, port=port)
        assert push_res["status"] == "ok"
        assert push_res["pushed_notes"] == 1
        assert received[1]["params"]["pitch"] == 33


@pytest.mark.asyncio
async def test_musescore_live_push_blocks_xxe_doctype(tmp_path: Path) -> None:
    bad_xml = tmp_path / "xxe.musicxml"
    bad_xml.write_text(
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><score-partwise/>',
        encoding="utf-8",
    )
    with pytest.raises(SecurityValidationError):
        await musescore_live_push(str(bad_xml), port=8765)
