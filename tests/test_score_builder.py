"""Tests for 4-string bass fretboard assignment, chord inference, MusicXML, and MIDI."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from bass_mcp.score_builder import (
    assign_bass_tab_positions,
    infer_measure_chords,
    render_ascii_bass_tab,
    write_midi_file,
    write_musicxml_score,
)
from bass_mcp.transcriber import BassNote


def test_assign_bass_tab_positions_maps_open_strings_and_minimizes_shifts() -> None:
    notes = [
        BassNote(28, "E1", 0.0, 0.5),  # String 4, Fret 0
        BassNote(31, "G1", 0.5, 0.5),  # String 4, Fret 3
        BassNote(33, "A1", 1.0, 0.5),  # String 3, Fret 0
        BassNote(35, "B1", 1.5, 0.5),  # String 3, Fret 2
    ]
    tabbed = assign_bass_tab_positions(notes, string_count=4)
    assert [(t.string_num, t.fret) for t in tabbed] == [(4, 0), (4, 3), (3, 0), (3, 2)]


def test_infer_measure_chords_major_and_minor() -> None:
    # Measure 1 (0..2s at 120 BPM): E1, G#1, B1 -> E major
    # Measure 2 (2..4s at 120 BPM): A1, C2, E2 -> A minor
    notes = [
        BassNote(28, "E1", 0.0, 0.5),
        BassNote(32, "G#1", 0.5, 0.5),
        BassNote(35, "B1", 1.0, 1.0),
        BassNote(33, "A1", 2.0, 0.5),
        BassNote(36, "C2", 2.5, 0.5),
        BassNote(40, "E2", 3.0, 1.0),
    ]
    tabbed = assign_bass_tab_positions(notes)
    chords = infer_measure_chords(tabbed, bpm=120.0)
    assert [c.symbol for c in chords] == ["E", "Am"]


def test_write_musicxml_and_midi_escapes_xml_injection(tmp_path: Path) -> None:
    notes = [
        BassNote(28, "E1", 0.0, 0.5),
        BassNote(31, "G1", 0.5, 0.5),
        BassNote(33, "A1", 1.0, 0.5),
        BassNote(35, "B1", 1.5, 0.5),
    ]
    tabbed = assign_bass_tab_positions(notes)
    chords = infer_measure_chords(tabbed, bpm=120.0)

    malicious_title = '<script>alert("xss")</script> & Bass'
    xml_path = tmp_path / "score.musicxml"
    mid_path = tmp_path / "score.mid"

    write_musicxml_score(tabbed, chords, xml_path, title=malicious_title, bpm=120.0)
    write_midi_file(tabbed, mid_path, bpm=120.0)

    raw_xml = xml_path.read_text(encoding="utf-8")
    assert "<script>" not in raw_xml
    assert "&lt;script&gt;" in raw_xml

    tree = ET.parse(xml_path)
    root = tree.getroot()
    assert root.tag == "score-partwise"
    assert root.findtext("./work/work-title") == malicious_title
    frets = [el.text for el in root.findall(".//technical/fret")]
    assert frets == ["0", "3", "0", "2"]

    midi_bytes = mid_path.read_bytes()
    assert midi_bytes.startswith(b"MThd")
    assert b"MTrk" in midi_bytes

    ascii_tab = render_ascii_bass_tab(tabbed, chords, bpm=120.0)
    assert "G|" in ascii_tab and "E|" in ascii_tab
    assert "Em" in ascii_tab or "E" in ascii_tab
