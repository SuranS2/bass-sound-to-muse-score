"""Tests for 4-string bass fretboard assignment, chord inference, MusicXML, and MIDI."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from bass_mcp.score_builder import (
    _quantize_notes_to_grid,
    assign_bass_tab_positions,
    infer_measure_chords,
    normalize_note_onsets,
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


def test_assign_bass_tab_positions_spyair_chorus_horizontal() -> None:
    pitches = [36, 36, 36, 36, 32, 32, 32, 32, 39, 39, 39, 39, 38, 38, 38, 38]
    notes = [BassNote(p, "N", i * 0.265, 0.22) for i, p in enumerate(pitches)]
    tabbed = assign_bass_tab_positions(notes, string_count=4)
    coords = [(t.string_num, t.fret) for t in tabbed]
    assert coords == [(3, 3)] * 4 + [(4, 4)] * 4 + [(3, 6)] * 4 + [(3, 5)] * 4


def test_assign_bass_tab_positions_octave_shapes_prechorus() -> None:
    pitches = [29, 41, 31, 43, 32, 44, 34, 46]
    notes = [BassNote(p, "N", i * 0.265, 0.20) for i, p in enumerate(pitches)]
    tabbed = assign_bass_tab_positions(notes, string_count=4)
    coords = [(t.string_num, t.fret) for t in tabbed]
    assert coords == [
        (4, 1),
        (2, 3),
        (4, 3),
        (2, 5),
        (4, 4),
        (2, 6),
        (4, 6),
        (2, 8),
    ]


def test_assign_bass_tab_positions_intro_and_bridge() -> None:
    intro_pitches = [32, 31, 43, 31, 36, 39]
    intro_notes = [BassNote(p, "N", i * 0.265, 0.20) for i, p in enumerate(intro_pitches)]
    intro_tab = [(t.string_num, t.fret) for t in assign_bass_tab_positions(intro_notes, 4)]
    assert intro_tab == [(4, 4), (4, 3), (2, 5), (4, 3), (3, 3), (3, 6)]

    bridge_pitches = [34, 34, 35, 35, 37, 37]
    bridge_notes = [BassNote(p, "N", i * 0.265, 0.20) for i, p in enumerate(bridge_pitches)]
    bridge_tab = [(t.string_num, t.fret) for t in assign_bass_tab_positions(bridge_notes, 4)]
    assert bridge_tab[2:4] == [(4, 7), (4, 7)]
    assert all(f >= 4 for _, f in bridge_tab[4:6])


def test_quantize_notes_prevents_16th_rest_fragmentation(tmp_path: Path) -> None:
    # Leading 9.067s intro silence + 8 plucked 8th notes with 0.18s sustain at 113 BPM
    raw_notes = [
        BassNote(36, "C2", 9.067 + i * 0.266, 0.18)
        for i in range(8)
    ]
    aligned = normalize_note_onsets(raw_notes)
    assert aligned[0].onset_sec == 0.0

    tabbed = assign_bass_tab_positions(aligned, string_count=4)
    total_measures, items = _quantize_notes_to_grid(tabbed, bpm=113.0, beats_per_measure=4)
    assert total_measures == 1
    assert [(it["start_div"], it["dur_div"]) for it in items] == [(i * 2, 2) for i in range(8)]

    xml_path = tmp_path / "clean_8ths.musicxml"
    chords = infer_measure_chords(tabbed, bpm=113.0)
    write_musicxml_score(tabbed, chords, xml_path, bpm=113.0)
    root = ET.parse(xml_path).getroot()
    staff1_notes = [n for n in root.findall(".//measure[@number='1']/note") if n.findtext("staff") == "1"]
    assert all(n.find("rest") is None for n in staff1_notes)
    assert [n.findtext("type") for n in staff1_notes] == ["eighth"] * 8


def test_normalize_note_onsets_preserves_intro_bars_for_ensemble(tmp_path: Path) -> None:
    raw_notes = [BassNote(32, "G#1", 9.056 + i * 0.2655, 0.18) for i in range(4)]
    aligned = normalize_note_onsets(raw_notes, bpm=113.0, preserve_intro_bars=True)
    measure_sec = 4 * 60.0 / 113.0
    assert abs(aligned[0].onset_sec - 4.0 * measure_sec) < 0.01

    tabbed = assign_bass_tab_positions(aligned, string_count=4)
    chords = infer_measure_chords(tabbed, bpm=113.0)
    xml_path = tmp_path / "with_intro.musicxml"
    write_musicxml_score(tabbed, chords, xml_path, bpm=113.0)
    root = ET.parse(xml_path).getroot()
    for m_num in ("1", "2", "3", "4"):
        m_notes = root.findall(f".//measure[@number='{m_num}']/note")
        assert all(n.find("rest") is not None for n in m_notes)
    m5_notes = [n for n in root.findall(".//measure[@number='5']/note") if n.findtext("staff") == "2"]
    assert any(n.find("rest") is None for n in m5_notes)


def test_infer_key_fifths_and_half_measure_chords_use_flats(tmp_path: Path) -> None:
    beat = 60.0 / 113.0
    notes = [
        BassNote(32, "G#1", 0.0 * beat, 0.4, 85),
        BassNote(32, "G#1", 0.5 * beat, 0.4, 85),
        BassNote(31, "G1", 2.0 * beat, 0.4, 85),
        BassNote(43, "G2", 3.0 * beat, 0.4, 85),
        BassNote(36, "C2", 4.0 * beat, 0.8, 85),
        BassNote(39, "D#2", 6.0 * beat, 0.8, 85),
    ]
    tabbed = assign_bass_tab_positions(notes, string_count=4)
    chords = infer_measure_chords(tabbed, bpm=113.0)
    assert [c.symbol for c in chords] == ["Ab", "Gm", "Cm", "Eb"]
    assert [c.beat_offset for c in chords] == [0, 2, 0, 2]

    xml_path = tmp_path / "flat_chords.musicxml"
    write_musicxml_score(tabbed, chords, xml_path, bpm=113.0)
    root = ET.parse(xml_path).getroot()
    assert int(root.findtext(".//attributes/key/fifths")) < 0
    m1_offsets = [h.findtext("offset") for h in root.findall(".//measure[@number='1']/harmony")]
    assert m1_offsets == [None, "8"]


def test_quantize_recovers_odd_16th_rest_jitter_in_fast_eight_beat() -> None:
    from bass_mcp.score_builder import _quantize_notes_to_grid

    bpm = 172.0
    eighth = 30.0 / bpm
    # Beat 3 (slot 4) clean, beat 3+ (slot 6) omitted -> late onset lands on odd 16th slot 9 (4.5 beats? slot 8+1)
    notes = assign_bass_tab_positions(
        [
            BassNote(29, "F1", 0 * eighth, 0.17, 85),
            BassNote(29, "F1", 1 * eighth, 0.16, 80),
            BassNote(29, "F1", 2 * eighth, 0.17, 82),
            BassNote(29, "F1", 3 * eighth, 0.16, 79),
            BassNote(29, "F1", 4.55 * eighth, 0.11, 82),  # lands at div 9 instead of div 8
            BassNote(29, "F1", 5 * eighth, 0.17, 84),     # div 10
            BassNote(29, "F1", 6 * eighth, 0.17, 84),     # div 12
            BassNote(29, "F1", 7 * eighth, 0.16, 78),     # div 14
        ],
        string_count=4,
    )
    _, grid = _quantize_notes_to_grid(notes, bpm=bpm, beats_per_measure=4)
    assert [(it["start_div"], it["dur_div"]) for it in grid] == [
        (0, 2), (2, 2), (4, 2), (6, 2), (8, 2), (10, 2), (12, 2), (14, 2)
    ]


def test_write_musicxml_embeds_escaped_synced_lyrics(tmp_path: Path) -> None:
    from bass_mcp.metadata import LyricLine

    notes = assign_bass_tab_positions(
        [BassNote(29, "F1", i * 0.35, 0.30, 85) for i in range(8)],
        string_count=4,
    )
    chords = infer_measure_chords(notes, bpm=172.0)
    lyrics = [
        LyricLine(0.0, "I am <standing>"),
        LyricLine(1.4, "in the line & holding"),
    ]
    xml_path = tmp_path / "with_lyrics.musicxml"
    write_musicxml_score(notes, chords, xml_path, bpm=172.0, lyrics=lyrics)
    raw = xml_path.read_text(encoding="utf-8")
    assert "&lt;standing&gt;" in raw
    root = ET.parse(xml_path).getroot()
    lyric_texts = [el.text for el in root.findall(".//lyric/text")]
    assert any("I am <standing>" in (t or "") for t in lyric_texts)
    assert any("in the line" in (t or "") for t in lyric_texts)
