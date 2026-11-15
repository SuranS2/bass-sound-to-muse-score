"""4-string/5-string Bass TAB solver, chord inference, MusicXML 4.0, MIDI, and MuseScore CLI."""

from __future__ import annotations

import importlib.util
import os
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from bass_mcp.security import SecurityValidationError, run_safe_command
from bass_mcp.transcriber import BassNote

DIVISIONS_PER_QUARTER = 4  # 16th-note grid

# MusicXML string 1 = highest-pitched string (G), string N = lowest-pitched string (E or B)
STRINGS_4: tuple[tuple[int, int, str, int], ...] = (
    (1, 43, "G", 2),
    (2, 38, "D", 2),
    (3, 33, "A", 1),
    (4, 28, "E", 1),
)
STRINGS_5: tuple[tuple[int, int, str, int], ...] = (
    (1, 43, "G", 2),
    (2, 38, "D", 2),
    (3, 33, "A", 1),
    (4, 28, "E", 1),
    (5, 23, "B", 0),
)
MAX_FRET = 24

_STEP_ALTER_SHARP = (
    ("C", 0),
    ("C", 1),
    ("D", 0),
    ("D", 1),
    ("E", 0),
    ("F", 0),
    ("F", 1),
    ("G", 0),
    ("G", 1),
    ("A", 0),
    ("A", 1),
    ("B", 0),
)
_STEP_ALTER_FLAT = (
    ("C", 0),
    ("D", -1),
    ("D", 0),
    ("E", -1),
    ("E", 0),
    ("F", 0),
    ("G", -1),
    ("G", 0),
    ("A", -1),
    ("A", 0),
    ("B", -1),
    ("B", 0),
)
_PC_LABELS_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_PC_LABELS_FLAT = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_MAJOR_SCALE_INTERVALS = (0, 2, 4, 5, 7, 9, 11)
# Circle-of-fifths major pitch classes keyed by fifths (-5..5)
_FIFTHS_TO_MAJOR_PC: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 7),
    (-1, 5),
    (2, 2),
    (-2, 10),
    (3, 9),
    (-3, 3),
    (4, 4),
    (-4, 8),
    (5, 11),
    (-5, 1),
)

_DURATION_TOKENS: tuple[tuple[int, str, bool], ...] = (
    (16, "whole", False),
    (12, "half", True),
    (8, "half", False),
    (6, "quarter", True),
    (4, "quarter", False),
    (3, "eighth", True),
    (2, "eighth", False),
    (1, "16th", False),
)


@dataclass(frozen=True)
class TabbedBassNote:
    """Bass note enriched with playable string and fret assignment."""

    pitch_midi: int
    note_name: str
    onset_sec: float
    duration_sec: float
    velocity: int
    string_num: int
    fret: int
    pitch_bend: int = 0


@dataclass(frozen=True)
class MeasureChord:
    """Inferred chord symbol for a measure (or half-measure)."""

    measure_num: int
    root_step: str
    root_alter: int
    quality_suffix: str
    musicxml_kind: str
    symbol: str
    beat_offset: int = 0


def _strings_for_count(string_count: int) -> tuple[tuple[int, int, str, int], ...]:
    if string_count == 4:
        return STRINGS_4
    if string_count == 5:
        return STRINGS_5
    raise SecurityValidationError("string_count must be 4 or 5.")


def _fit_pitch_to_bass_range(pitch_midi: int, min_midi: int, max_midi: int) -> int:
    p = int(pitch_midi)
    while p < min_midi:
        p += 12
    while p > max_midi:
        p -= 12
    return max(min_midi, min(max_midi, p))


def compute_onset_shift_sec(
    notes: list[BassNote],
    min_trim_sec: float = 0.25,
    bpm: float | None = None,
    beats_per_measure: int = 4,
    preserve_intro_bars: bool = False,
) -> float:
    """Return leading time shift (seconds) used to align note onsets to bar boundaries."""
    if not notes:
        return 0.0
    t0 = min(n.onset_sec for n in notes)
    if preserve_intro_bars and bpm and bpm > 0:
        measure_sec = beats_per_measure * (60.0 / max(bpm, 20.0))
        if t0 < 0.75 * measure_sec:
            return t0 if t0 > min_trim_sec else 0.0
        bars_raw = t0 / measure_sec
        frac = bars_raw - int(bars_raw)
        if frac <= 0.40 or frac >= 0.85:
            intro_bars = max(0, int(round(bars_raw)))
            return t0 - intro_bars * measure_sec
        return 0.0
    return t0 if t0 > min_trim_sec else 0.0


def normalize_note_onsets(
    notes: list[BassNote],
    min_trim_sec: float = 0.25,
    bpm: float | None = None,
    beats_per_measure: int = 4,
    preserve_intro_bars: bool = False,
) -> list[BassNote]:
    """Shift leading silence while optionally preserving integer intro rest bars for ensemble sync."""
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: n.onset_sec)
    shift = compute_onset_shift_sec(
        ordered,
        min_trim_sec=min_trim_sec,
        bpm=bpm,
        beats_per_measure=beats_per_measure,
        preserve_intro_bars=preserve_intro_bars,
    )
    if abs(shift) < 1e-4:
        return ordered
    return [
        BassNote(
            pitch_midi=n.pitch_midi,
            note_name=n.note_name,
            onset_sec=round(max(0.0, n.onset_sec - shift), 4),
            duration_sec=n.duration_sec,
            velocity=n.velocity,
            pitch_bend=n.pitch_bend,
        )
        for n in ordered
    ]


def _position_cost(s_num: int, fret: int, pitch_midi: int, has_open_e_context: bool) -> float:
    """Static fretboard cost favoring middle-neck 3rd/4th-string box positions over thin low frets."""
    if fret > 12:
        return 3.0 + (fret - 12) * 0.9

    if fret == 0:
        if pitch_midi == 28:  # E1 is only playable open on 4-string bass
            return 0.0
        if s_num in (1, 2):
            return 4.2
        if s_num == 3:
            return 0.0 if has_open_e_context else 2.8
        return 0.2

    if s_num in (1, 2) and fret in (1, 2):
        return 3.2 + (0.16 * fret)
    if s_num == 3 and fret == 1:
        return 1.8 + (0.12 * fret)
    if s_num == 3 and fret == 2:
        return (0.3 if has_open_e_context else 1.2) + (0.12 * fret)

    string_weight = 0.25 if s_num == 1 else (0.16 if s_num == 2 else 0.12)
    return fret * string_weight


def _transition_cost(
    s_prev: int,
    f_prev: int,
    m_prev: int,
    s_cur: int,
    f_cur: int,
    m_cur: int,
    has_open_e_context: bool,
) -> float:
    """Fretboard transition cost enforcing same-fret repeats, 2-string octave shapes, and box continuity."""
    if m_cur == m_prev:
        return 0.0 if (s_cur == s_prev and f_cur == f_prev) else 6.0

    # Standard bass octave shape: +2 frets, 2 strings higher in pitch (lower string_num).
    if abs(m_cur - m_prev) == 12:
        is_up_octave = (m_cur - m_prev == 12) and (s_prev - s_cur == 2) and (f_cur - f_prev == 2)
        is_down_octave = (m_prev - m_cur == 12) and (s_cur - s_prev == 2) and (f_prev - f_cur == 2)
        if is_up_octave or is_down_octave:
            return 0.0

    df = abs(f_cur - f_prev)
    ds = abs(s_cur - s_prev)

    if has_open_e_context and (f_cur == 0 or f_prev == 0) and max(f_cur, f_prev) <= 4:
        cost = min(df, 1) * 0.35 + ds * 0.25
    elif df <= 3:
        cost = (df * 0.45) + (ds * 0.35)
        if s_cur == s_prev and s_cur in (3, 4):
            cost = max(0.0, cost - 0.35)
    else:
        cost = 1.35 + ((df - 3) * 1.35) + (ds * 0.45)

    if abs(m_cur - m_prev) == 12:
        cost += 3.5
    return cost


def assign_bass_tab_positions(
    notes: list[BassNote],
    string_count: int = 4,
) -> list[TabbedBassNote]:
    """Assign optimal (string, fret) coordinates minimizing hand shifts across the fretboard."""
    strings = _strings_for_count(string_count)
    if not notes:
        return []

    min_midi = strings[-1][1]
    max_midi = strings[0][1] + MAX_FRET

    candidates_per_note: list[list[tuple[int, int, int]]] = []
    for note in notes:
        p = _fit_pitch_to_bass_range(note.pitch_midi, min_midi, max_midi)
        opts = [
            (s_num, p - open_midi, p)
            for s_num, open_midi, _, _ in strings
            if 0 <= (p - open_midi) <= MAX_FRET
        ]
        if not opts:
            opts = [(strings[-1][0], 0, strings[-1][1])]
        candidates_per_note.append(opts)

    n = len(candidates_per_note)
    open_e_mask = [
        any(notes[j].pitch_midi == 28 for j in range(max(0, idx - 3), min(n, idx + 4)))
        for idx in range(n)
    ]

    # Dynamic programming over fretboard positions.
    dp_cost: list[list[float]] = []
    dp_prev: list[list[int]] = []

    first_costs = [
        _position_cost(s_num, fret, fitted_midi, open_e_mask[0])
        for s_num, fret, fitted_midi in candidates_per_note[0]
    ]
    dp_cost.append(first_costs)
    dp_prev.append([-1] * len(first_costs))

    for idx in range(1, n):
        cur_opts = candidates_per_note[idx]
        prev_opts = candidates_per_note[idx - 1]
        cur_costs: list[float] = []
        cur_back: list[int] = []
        ctx_open_e = open_e_mask[idx] or open_e_mask[idx - 1]
        for s_cur, f_cur, m_cur in cur_opts:
            base_cost = _position_cost(s_cur, f_cur, m_cur, open_e_mask[idx])
            best_total = float("inf")
            best_prev = 0
            for p_idx, (s_prev, f_prev, m_prev) in enumerate(prev_opts):
                trans = _transition_cost(
                    s_prev, f_prev, m_prev, s_cur, f_cur, m_cur, ctx_open_e
                )
                total = dp_cost[idx - 1][p_idx] + base_cost + trans
                if total < best_total:
                    best_total = total
                    best_prev = p_idx
            cur_costs.append(best_total)
            cur_back.append(best_prev)
        dp_cost.append(cur_costs)
        dp_prev.append(cur_back)

    chosen_indices = [0] * n
    chosen_indices[-1] = min(range(len(dp_cost[-1])), key=lambda k: dp_cost[-1][k])
    for idx in range(n - 2, -1, -1):
        chosen_indices[idx] = dp_prev[idx + 1][chosen_indices[idx + 1]]

    tabbed: list[TabbedBassNote] = []
    for idx, note in enumerate(notes):
        s_num, fret, fitted_midi = candidates_per_note[idx][chosen_indices[idx]]
        tabbed.append(
            TabbedBassNote(
                pitch_midi=fitted_midi,
                note_name=note.note_name,
                onset_sec=note.onset_sec,
                duration_sec=note.duration_sec,
                velocity=note.velocity,
                string_num=s_num,
                fret=fret,
                pitch_bend=note.pitch_bend,
            )
        )
    return tabbed


def infer_key_fifths(notes: list[TabbedBassNote] | list[BassNote]) -> int:
    """Infer MusicXML circle-of-fifths key signature (-5..5) from weighted pitch classes."""
    if not notes:
        return 0
    pc_weights = [0.0] * 12
    for n in notes:
        pc_weights[n.pitch_midi % 12] += max(0.08, n.duration_sec) * (0.5 + n.velocity / 127.0)
    total = sum(pc_weights)
    if total <= 0:
        return 0

    best_fifths = 0
    best_score = -1.0
    for fifths, major_pc in _FIFTHS_TO_MAJOR_PC:
        scale_pcs = {(major_pc + interval) % 12 for interval in _MAJOR_SCALE_INTERVALS}
        score = sum(pc_weights[pc] for pc in scale_pcs) - 0.002 * abs(fifths) * total
        if score > best_score:
            best_score = score
            best_fifths = fifths
    return best_fifths


def _diatonic_pcs_for_fifths(key_fifths: int) -> set[int]:
    for fifths, major_pc in _FIFTHS_TO_MAJOR_PC:
        if fifths == key_fifths:
            return {(major_pc + interval) % 12 for interval in _MAJOR_SCALE_INTERVALS}
    return set(_MAJOR_SCALE_INTERVALS)


def _build_chord_for_slice(
    m_idx: int,
    m_notes: list[TabbedBassNote],
    measure_start_sec: float,
    sec_per_beat: float,
    beat_offset: int,
    use_flats: bool,
    diatonic_pcs: set[int] | None = None,
) -> MeasureChord:
    pc_weights: dict[int, float] = {}
    for n in m_notes:
        beat_rel = (n.onset_sec - measure_start_sec) / sec_per_beat - beat_offset
        strong_bonus = 3.5 if beat_rel < 0.5 else (1.3 if abs(beat_rel - 2.0) < 0.5 else 1.0)
        pc = n.pitch_midi % 12
        pc_weights[pc] = pc_weights.get(pc, 0.0) + max(0.1, n.duration_sec) * strong_bonus

    root_pc = max(pc_weights, key=lambda k: pc_weights[k])
    intervals = {(n.pitch_midi % 12 - root_pc) % 12 for n in m_notes}
    if 3 in intervals and 4 not in intervals:
        suffix, kind = ("m7", "minor-seventh") if 10 in intervals else ("m", "minor")
    elif 4 in intervals:
        suffix, kind = ("7", "dominant") if 10 in intervals else (("maj7", "major-seventh") if 11 in intervals else ("", "major"))
    elif 10 in intervals:
        suffix, kind = "7", "dominant"
    elif 11 in intervals:
        suffix, kind = "maj7", "major-seventh"
    elif diatonic_pcs and ((root_pc + 3) % 12 in diatonic_pcs) and ((root_pc + 4) % 12 not in diatonic_pcs):
        suffix, kind = "m", "minor"
    else:
        suffix, kind = "", "major"

    step, alter = (_STEP_ALTER_FLAT if use_flats else _STEP_ALTER_SHARP)[root_pc]
    label = (_PC_LABELS_FLAT if use_flats else _PC_LABELS_SHARP)[root_pc]
    return MeasureChord(m_idx, step, alter, suffix, kind, f"{label}{suffix}", beat_offset=beat_offset)


def infer_measure_chords(
    notes: list[TabbedBassNote],
    bpm: float = 120.0,
    beats_per_measure: int = 4,
    key_fifths: int | None = None,
) -> list[MeasureChord]:
    """Infer root-and-quality chord symbols per measure (splitting half-bars when root changes)."""
    if not notes:
        return [MeasureChord(1, "E", 0, "", "major", "E")]

    inferred_fifths = infer_key_fifths(notes) if key_fifths is None else key_fifths
    use_flats = inferred_fifths < 0
    diatonic_pcs = _diatonic_pcs_for_fifths(inferred_fifths)
    sec_per_beat = 60.0 / max(bpm, 20.0)
    measure_sec = beats_per_measure * sec_per_beat
    half_beats = beats_per_measure // 2
    max_end = max(n.onset_sec + n.duration_sec for n in notes)
    total_measures = max(1, int((max_end - 1e-6) // measure_sec) + 1)

    buckets: dict[int, list[TabbedBassNote]] = {m: [] for m in range(1, total_measures + 1)}
    for n in notes:
        m_idx = min(total_measures, max(1, int(n.onset_sec // measure_sec) + 1))
        buckets[m_idx].append(n)

    chords: list[MeasureChord] = []
    last_chord: MeasureChord | None = None

    for m_idx in range(1, total_measures + 1):
        m_notes = buckets[m_idx]
        if not m_notes:
            if last_chord is not None:
                chords.append(
                    MeasureChord(
                        m_idx,
                        last_chord.root_step,
                        last_chord.root_alter,
                        last_chord.quality_suffix,
                        last_chord.musicxml_kind,
                        last_chord.symbol,
                    )
                )
            continue

        m_start = (m_idx - 1) * measure_sec
        if beats_per_measure >= 4:
            split_sec = m_start + max(0.5, half_beats - 0.55) * sec_per_beat
            first_half = [n for n in m_notes if n.onset_sec < split_sec]
            second_half = [n for n in m_notes if n.onset_sec >= split_sec]
            if first_half and second_half:
                c1 = _build_chord_for_slice(m_idx, first_half, m_start, sec_per_beat, 0, use_flats, diatonic_pcs)
                c2 = _build_chord_for_slice(m_idx, second_half, m_start, sec_per_beat, half_beats, use_flats, diatonic_pcs)
                first_pcs = {n.pitch_midi % 12 for n in first_half}
                second_pcs = {n.pitch_midi % 12 for n in second_half}
                w1 = sum(max(0.1, n.duration_sec) for n in first_half)
                w2 = sum(max(0.1, n.duration_sec) for n in second_half)
                r1 = max({n.pitch_midi % 12 for n in first_half}, key=lambda pc: sum(max(0.1, n.duration_sec) for n in first_half if n.pitch_midi % 12 == pc))
                r2 = max({n.pitch_midi % 12 for n in second_half}, key=lambda pc: sum(max(0.1, n.duration_sec) for n in second_half if n.pitch_midi % 12 == pc))
                is_arpeggio_fifth = len(first_pcs) >= 2 and len(second_pcs) == 1 and ((r2 - r1) % 12 in (0, 3, 4, 7))
                if r1 != r2 and not is_arpeggio_fifth and w2 >= 0.35 * (w1 + w2):
                    chords.extend((c1, c2))
                    last_chord = c2
                    continue

        chord = _build_chord_for_slice(m_idx, m_notes, m_start, sec_per_beat, 0, use_flats, diatonic_pcs)
        chords.append(chord)
        last_chord = chord

    return chords or [MeasureChord(1, "E", 0, "", "major", "E")]


def _encode_vlq(value: int) -> bytes:
    value = max(0, int(value))
    chunks = [value & 0x7F]
    value >>= 7
    while value:
        chunks.append((value & 0x7F) | 0x80)
        value >>= 7
    chunks.reverse()
    return bytes(chunks)


def write_midi_file(notes: list[TabbedBassNote], output_path: Path, bpm: float = 120.0) -> Path:
    """Write a standard MIDI file (.mid) with Electric Bass program 33."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("pretty_midi") is not None:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
        bass = pretty_midi.Instrument(program=33, name="Electric Bass")
        for n in notes:
            bass.notes.append(
                pretty_midi.Note(
                    velocity=int(max(1, min(127, n.velocity))),
                    pitch=int(max(0, min(127, n.pitch_midi))),
                    start=float(max(0.0, n.onset_sec)),
                    end=float(max(n.onset_sec + 0.05, n.onset_sec + n.duration_sec)),
                )
            )
        pm.instruments.append(bass)
        pm.write(str(output_path))
        return output_path

    # Pure Python Standard MIDI Format 0 fallback (480 ticks per quarter note).
    ppq = 480
    us_per_quarter = int(round(60_000_000 / max(bpm, 20.0)))
    events: list[tuple[int, int, bytes]] = [
        (0, 0, b"\xFF\x51\x03" + us_per_quarter.to_bytes(3, "big")),
        (0, 1, bytes([0xC0, 33])),  # Program 33: Electric Bass (finger)
    ]
    for n in notes:
        start_tick = max(0, int(round(n.onset_sec * (bpm / 60.0) * ppq)))
        dur_ticks = max(24, int(round(n.duration_sec * (bpm / 60.0) * ppq)))
        pitch = max(0, min(127, int(n.pitch_midi)))
        vel = max(1, min(127, int(n.velocity)))
        events.append((start_tick, 2, bytes([0x90, pitch, vel])))
        events.append((start_tick + dur_ticks, 1, bytes([0x80, pitch, 0])))

    events.sort(key=lambda item: (item[0], item[1]))
    track_bytes = bytearray()
    prev_tick = 0
    for tick, _, payload in events:
        delta = tick - prev_tick
        prev_tick = tick
        track_bytes.extend(_encode_vlq(delta))
        track_bytes.extend(payload)
    track_bytes.extend(b"\x00\xFF\x2F\x00")  # End of track

    header = struct.pack(">4sIHHH", b"MThd", 6, 0, 1, ppq)
    track = struct.pack(">4sI", b"MTrk", len(track_bytes)) + bytes(track_bytes)
    output_path.write_bytes(header + track)
    return output_path


def _decompose_divisions(duration_div: int) -> list[tuple[int, str, bool]]:
    """Split a duration in 16th-note divisions into standard MusicXML note types."""
    rem = max(1, duration_div)
    parts: list[tuple[int, str, bool]] = []
    while rem > 0:
        for val, note_type, dotted in _DURATION_TOKENS:
            if val <= rem:
                parts.append((val, note_type, dotted))
                rem -= val
                break
    return parts


def _quantize_notes_to_grid(
    notes: list[TabbedBassNote],
    bpm: float,
    beats_per_measure: int,
) -> tuple[int, list[dict]]:
    """Snap notes to a non-overlapping 16th-note grid with 8th-note rhythm stabilization."""
    measure_div = beats_per_measure * DIVISIONS_PER_QUARTER
    div_sec = (60.0 / max(bpm, 20.0)) / DIVISIONS_PER_QUARTER

    if not notes:
        return 1, []

    ordered = sorted(notes, key=lambda item: item.onset_sec)
    grid_items: list[dict] = []
    for idx, n in enumerate(ordered):
        raw_start = max(0.0, n.onset_sec / div_sec)
        start_div = max(0, int(round(raw_start)))
        prev_ioi = (n.onset_sec - ordered[idx - 1].onset_sec) if idx > 0 else float("inf")
        next_ioi = (ordered[idx + 1].onset_sec - n.onset_sec) if idx + 1 < len(ordered) else float("inf")
        even_ioi_ok = min(prev_ioi, next_ioi) >= 0.68 * (2.0 * div_sec) or (
            (prev_ioi + next_ioi) >= 1.75 * (2.0 * div_sec)
            and min(prev_ioi, next_ioi) >= 0.60 * (2.0 * div_sec)
        )
        if start_div % 2 == 1 and even_ioi_ok:
            candidates = [c for c in (start_div - 1, start_div + 1) if c >= 0]
            even_div = min(candidates, key=lambda c: abs(raw_start - c))
            if abs(raw_start - even_div) <= 0.68:
                start_div = even_div

        dur_div = max(1, int(round(n.duration_sec / div_sec)))
        if grid_items and start_div <= grid_items[-1]["start_div"]:
            continue
        if grid_items:
            prev = grid_items[-1]
            if prev["start_div"] + prev["dur_div"] > start_div:
                prev["dur_div"] = max(1, start_div - prev["start_div"])
        grid_items.append({"start_div": start_div, "dur_div": dur_div, "note": n})

    # Recover late-triggered 8th notes that landed on odd 16th slots leaving an empty preceding 16th rest.
    for idx, item in enumerate(grid_items):
        if item["start_div"] % 2 == 1 and item["dur_div"] == 1:
            prev_item = grid_items[idx - 1] if idx > 0 else None
            prev_start = prev_item["start_div"] if prev_item else -99
            next_start = grid_items[idx + 1]["start_div"] if idx + 1 < len(grid_items) else None
            if prev_start < item["start_div"] - 1 and next_start == item["start_div"] + 1:
                if prev_item and prev_item["start_div"] + prev_item["dur_div"] > item["start_div"] - 1:
                    prev_item["dur_div"] = max(1, (item["start_div"] - 1) - prev_item["start_div"])
                item["start_div"] -= 1
                item["dur_div"] = 2

    # Stabilize plucked 8th/quarter notes so acoustic decay does not fragment into 16th rests.
    for idx, item in enumerate(grid_items):
        if idx + 1 < len(grid_items):
            gap_to_next = grid_items[idx + 1]["start_div"] - item["start_div"]
        else:
            gap_to_next = measure_div - (item["start_div"] % measure_div)

        if item["dur_div"] == 1 and gap_to_next >= 2 and item["note"].duration_sec >= 1.05 * div_sec:
            item["dur_div"] = 2
        if (
            gap_to_next - item["dur_div"] == 1
            and item["start_div"] % 2 == 0
            and (item["dur_div"] + 1) % 2 == 0
        ):
            item["dur_div"] += 1

    last_end = max(it["start_div"] + it["dur_div"] for it in grid_items)
    total_measures = max(1, (last_end + measure_div - 1) // measure_div)
    return total_measures, grid_items


def _append_rest(measure_el: ET.Element, dur_div: int, staff_num: int) -> None:
    for part_div, note_type, dotted in _decompose_divisions(dur_div):
        note_el = ET.SubElement(measure_el, "note")
        ET.SubElement(note_el, "rest")
        ET.SubElement(note_el, "duration").text = str(part_div)
        ET.SubElement(note_el, "type").text = note_type
        if dotted:
            ET.SubElement(note_el, "dot")
        ET.SubElement(note_el, "staff").text = str(staff_num)


def _append_pitched_note(
    measure_el: ET.Element,
    note: TabbedBassNote,
    dur_div: int,
    staff_num: int,
    include_tab: bool,
    tie_start: bool = False,
    tie_stop: bool = False,
    use_flats: bool = False,
    lyric_text: str | None = None,
) -> None:
    parts = _decompose_divisions(dur_div)
    step, alter = (_STEP_ALTER_FLAT if use_flats else _STEP_ALTER_SHARP)[note.pitch_midi % 12]
    octave = (note.pitch_midi // 12) - 1

    for idx, (part_div, note_type, dotted) in enumerate(parts):
        is_first = idx == 0
        is_last = idx == len(parts) - 1
        start_tie = (tie_start and is_first) or (not is_last)
        stop_tie = (tie_stop and is_last) or (not is_first)

        note_el = ET.SubElement(measure_el, "note")
        pitch_el = ET.SubElement(note_el, "pitch")
        ET.SubElement(pitch_el, "step").text = step
        if alter != 0:
            ET.SubElement(pitch_el, "alter").text = str(alter)
        ET.SubElement(pitch_el, "octave").text = str(octave)

        ET.SubElement(note_el, "duration").text = str(part_div)
        if stop_tie:
            ET.SubElement(note_el, "tie", {"type": "stop"})
        if start_tie:
            ET.SubElement(note_el, "tie", {"type": "start"})

        ET.SubElement(note_el, "type").text = note_type
        if dotted:
            ET.SubElement(note_el, "dot")
        ET.SubElement(note_el, "staff").text = str(staff_num)

        if include_tab or start_tie or stop_tie:
            notations = ET.SubElement(note_el, "notations")
            if stop_tie:
                ET.SubElement(notations, "tied", {"type": "stop"})
            if start_tie:
                ET.SubElement(notations, "tied", {"type": "start"})
            if include_tab and is_first and not tie_stop:
                technical = ET.SubElement(notations, "technical")
                ET.SubElement(technical, "string").text = str(note.string_num)
                ET.SubElement(technical, "fret").text = str(note.fret)

        if lyric_text and is_first and not tie_stop and not include_tab:
            lyric_el = ET.SubElement(note_el, "lyric", {"number": "1"})
            ET.SubElement(lyric_el, "syllabic").text = "single"
            ET.SubElement(lyric_el, "text").text = lyric_text


def _split_lyric_chunks(text: str, max_words: int = 3, max_chars: int = 18) -> list[str]:
    words = text.strip().split()
    if not words:
        return []
    if len(words) == 1 and len(words[0]) <= max_chars:
        return words
    chunks: list[str] = []
    cur: list[str] = []
    for word in words:
        cand = " ".join([*cur, word])
        if cur and (len(cur) >= max_words or len(cand) > max_chars):
            chunks.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _map_lyrics_to_grid(
    grid_items: list[dict],
    measure_div: int,
    bpm: float,
    lyrics: list | None,
    time_shift_sec: float = 0.0,
) -> dict[int, str]:
    """Attach synced lyric chunks or plain lyric lines onto quantized note start_div positions."""
    if not grid_items or not lyrics:
        return {}

    div_sec = (60.0 / max(bpm, 20.0)) / DIVISIONS_PER_QUARTER
    synced: list[tuple[float, str]] = []
    plain: list[str] = []
    for entry in lyrics:
        if isinstance(entry, str):
            txt = entry.strip()
            if txt:
                plain.append(txt)
        elif isinstance(entry, dict):
            txt = str(entry.get("text") or "").strip()
            if txt and "onset_sec" in entry and entry["onset_sec"] is not None:
                synced.append((float(entry["onset_sec"]), txt))
            elif txt:
                plain.append(txt)
        elif hasattr(entry, "text"):
            txt = str(getattr(entry, "text") or "").strip()
            onset = getattr(entry, "onset_sec", None)
            if txt and onset is not None:
                synced.append((float(onset), txt))
            elif txt:
                plain.append(txt)

    assigned: dict[int, str] = {}
    if synced:
        synced.sort(key=lambda item: item[0])
        last_assigned_idx = -1
        lookahead_divs = max(8, int(round(3.0 / max(div_sec, 1e-4))))
        for line_idx, (onset_sec, line_text) in enumerate(synced):
            next_onset = synced[line_idx + 1][0] if line_idx + 1 < len(synced) else onset_sec + 4.0
            span_sec = min(max(next_onset - onset_sec, 1.0), 5.0)
            chunks = _split_lyric_chunks(line_text)
            if not chunks:
                continue
            aligned_start = max(0.0, onset_sec - time_shift_sec)
            for c_idx, chunk in enumerate(chunks):
                t_chunk = aligned_start + (c_idx / len(chunks)) * span_sec
                target_div = max(0, int(round(t_chunk / div_sec)))
                best_idx: int | None = None
                for g_idx in range(last_assigned_idx + 1, len(grid_items)):
                    s_div = grid_items[g_idx]["start_div"]
                    if s_div < target_div - 2:
                        continue
                    if s_div > target_div + lookahead_divs:
                        break
                    best_idx = g_idx
                    break
                if best_idx is not None:
                    assigned[grid_items[best_idx]["start_div"]] = chunk
                    last_assigned_idx = best_idx
        return assigned

    if plain:
        per_measure_first: dict[int, int] = {}
        for it in grid_items:
            m_num = (it["start_div"] // measure_div) + 1
            per_measure_first.setdefault(m_num, it["start_div"])
        active_measures = sorted(per_measure_first)
        for line_text, m_num in zip(plain, active_measures):
            assigned[per_measure_first[m_num]] = line_text[:36]
    return assigned


def write_musicxml_score(
    notes: list[TabbedBassNote],
    chords: list[MeasureChord],
    output_path: Path,
    title: str = "Bass Transcription",
    bpm: float = 120.0,
    string_count: int = 4,
    beats_per_measure: int = 4,
    lyrics: list | None = None,
    lyric_time_shift_sec: float = 0.0,
) -> Path:
    """Write a valid 2-staff (Bass Clef + 4/5-string TAB + Chord symbols) MusicXML 4.0 score."""
    strings = _strings_for_count(string_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    measure_div = beats_per_measure * DIVISIONS_PER_QUARTER
    total_measures, grid_items = _quantize_notes_to_grid(notes, bpm, beats_per_measure)
    total_measures = max(total_measures, max((c.measure_num for c in chords), default=0))
    key_fifths = infer_key_fifths(notes)
    use_flats = key_fifths < 0
    lyric_by_start_div = _map_lyrics_to_grid(
        grid_items,
        measure_div=measure_div,
        bpm=bpm,
        lyrics=lyrics,
        time_shift_sec=lyric_time_shift_sec,
    )

    # Split notes cleanly across bar boundaries.
    per_measure: dict[int, list[tuple[int, int, TabbedBassNote, bool, bool]]] = {
        m: [] for m in range(1, total_measures + 1)
    }
    for item in grid_items:
        cursor = item["start_div"]
        rem = item["dur_div"]
        note_obj = item["note"]
        first_chunk = True
        while rem > 0:
            m_idx = (cursor // measure_div) + 1
            offset_in_m = cursor % measure_div
            take = min(rem, measure_div - offset_in_m)
            has_more = rem > take
            if m_idx in per_measure:
                per_measure[m_idx].append(
                    (offset_in_m, take, note_obj, has_more, not first_chunk)
                )
            cursor += take
            rem -= take
            first_chunk = False

    root = ET.Element("score-partwise", {"version": "4.0"})
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = title.strip() or "Bass Transcription"

    ident = ET.SubElement(root, "identification")
    encoding = ET.SubElement(ident, "encoding")
    ET.SubElement(encoding, "software").text = "bass-sound-to-muse-score"

    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", {"id": "P1"})
    ET.SubElement(score_part, "part-name").text = "Electric Bass"
    score_inst = ET.SubElement(score_part, "score-instrument", {"id": "P1-I1"})
    ET.SubElement(score_inst, "instrument-name").text = "Electric Bass"
    midi_inst = ET.SubElement(score_part, "midi-instrument", {"id": "P1-I1"})
    ET.SubElement(midi_inst, "midi-channel").text = "1"
    ET.SubElement(midi_inst, "midi-program").text = "34"

    part = ET.SubElement(root, "part", {"id": "P1"})
    chords_by_measure: dict[int, list[MeasureChord]] = {}
    for c in chords:
        chords_by_measure.setdefault(c.measure_num, []).append(c)

    for m_idx in range(1, total_measures + 1):
        measure_el = ET.SubElement(part, "measure", {"number": str(m_idx)})
        if m_idx == 1:
            attrs = ET.SubElement(measure_el, "attributes")
            ET.SubElement(attrs, "divisions").text = str(DIVISIONS_PER_QUARTER)
            key = ET.SubElement(attrs, "key")
            ET.SubElement(key, "fifths").text = str(key_fifths)
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(beats_per_measure)
            ET.SubElement(time_el, "beat-type").text = "4"
            ET.SubElement(attrs, "staves").text = "2"

            clef1 = ET.SubElement(attrs, "clef", {"number": "1"})
            ET.SubElement(clef1, "sign").text = "F"
            ET.SubElement(clef1, "line").text = "4"

            clef2 = ET.SubElement(attrs, "clef", {"number": "2"})
            ET.SubElement(clef2, "sign").text = "TAB"
            ET.SubElement(clef2, "line").text = "5"

            staff_details = ET.SubElement(attrs, "staff-details", {"number": "2"})
            ET.SubElement(staff_details, "staff-lines").text = str(len(strings))
            # MusicXML staff-tuning line 1 is bottom string (E1 or B0), line N is top string (G2).
            for line_idx, (_, _, step, octave) in enumerate(reversed(strings), start=1):
                tuning = ET.SubElement(staff_details, "staff-tuning", {"line": str(line_idx)})
                ET.SubElement(tuning, "tuning-step").text = step
                ET.SubElement(tuning, "tuning-octave").text = str(octave)

            direction = ET.SubElement(measure_el, "direction", {"placement": "above"})
            dir_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(dir_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = str(int(round(bpm)))
            ET.SubElement(direction, "sound", {"tempo": str(round(bpm, 1))})

        for chord in chords_by_measure.get(m_idx, ()):
            harmony = ET.SubElement(measure_el, "harmony", {"placement": "above"})
            root_el = ET.SubElement(harmony, "root")
            ET.SubElement(root_el, "root-step").text = chord.root_step
            if chord.root_alter != 0:
                ET.SubElement(root_el, "root-alter").text = str(chord.root_alter)
            ET.SubElement(harmony, "kind").text = chord.musicxml_kind
            if chord.beat_offset > 0:
                ET.SubElement(harmony, "offset").text = str(chord.beat_offset * DIVISIONS_PER_QUARTER)

        events = sorted(per_measure[m_idx], key=lambda x: x[0])

        for staff_num, include_tab in ((1, False), (2, True)):
            if staff_num == 2:
                backup = ET.SubElement(measure_el, "backup")
                ET.SubElement(backup, "duration").text = str(measure_div)
            cursor = 0
            for start_off, dur_div, note_obj, tie_start, tie_stop in events:
                if start_off > cursor:
                    _append_rest(measure_el, start_off - cursor, staff_num)
                global_div = (m_idx - 1) * measure_div + start_off
                note_lyric = (
                    lyric_by_start_div.get(global_div)
                    if staff_num == 1 and not tie_stop
                    else None
                )
                _append_pitched_note(
                    measure_el,
                    note_obj,
                    dur_div,
                    staff_num=staff_num,
                    include_tab=include_tab,
                    tie_start=tie_start,
                    tie_stop=tie_stop,
                    use_flats=use_flats,
                    lyric_text=note_lyric,
                )
                cursor = start_off + dur_div
            if cursor < measure_div:
                _append_rest(measure_el, measure_div - cursor, staff_num)

    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    output_path.write_bytes(xml_bytes)
    return output_path


def render_ascii_bass_tab(
    notes: list[TabbedBassNote],
    chords: list[MeasureChord],
    bpm: float = 120.0,
    string_count: int = 4,
    beats_per_measure: int = 4,
    measures_per_row: int = 4,
) -> str:
    """Render readable 4/5-line ASCII Bass TAB with chord symbols above each measure."""
    strings = _strings_for_count(string_count)
    measure_div = beats_per_measure * DIVISIONS_PER_QUARTER
    total_measures, grid_items = _quantize_notes_to_grid(notes, bpm, beats_per_measure)
    total_measures = max(total_measures, max((c.measure_num for c in chords), default=0))
    chord_by_measure: dict[int, str] = {}
    for c in chords:
        chord_by_measure[c.measure_num] = (
            f"{chord_by_measure[c.measure_num]}/{c.symbol}"
            if c.measure_num in chord_by_measure
            else c.symbol
        )

    # Build per-measure string grids (16 columns per 4/4 measure, 2 chars per step).
    cell_width = 2
    measure_grids: dict[int, dict[int, list[str]]] = {
        m: {s_num: ["--" for _ in range(measure_div)] for s_num, _, _, _ in strings}
        for m in range(1, total_measures + 1)
    }

    for item in grid_items:
        pos = item["start_div"]
        m_idx = (pos // measure_div) + 1
        col = pos % measure_div
        note_obj: TabbedBassNote = item["note"]
        if m_idx in measure_grids:
            fret_str = str(note_obj.fret)
            measure_grids[m_idx][note_obj.string_num][col] = fret_str.ljust(cell_width, "-")

    rows: list[str] = []
    for row_start in range(1, total_measures + 1, measures_per_row):
        row_end = min(total_measures, row_start + measures_per_row - 1)
        chord_line = "  "
        for m_idx in range(row_start, row_end + 1):
            label = chord_by_measure.get(m_idx, "")
            seg_len = measure_div * cell_width + 1
            chord_line += label.ljust(seg_len)
        rows.append(chord_line.rstrip())

        for s_num, _, s_label, _ in strings:
            line = f"{s_label}|"
            for m_idx in range(row_start, row_end + 1):
                line += "".join(measure_grids[m_idx][s_num]) + "|"
            rows.append(line)
        rows.append("")

    return "\n".join(rows).rstrip()


def find_musescore_binary() -> str | None:
    """Locate a local MuseScore 4/3 executable if installed."""
    env_bin = os.environ.get("BASS_MCP_MUSESCORE_BIN", "").strip()
    if env_bin:
        candidate = Path(env_bin)
        if candidate.is_file() and not candidate.is_symlink():
            return str(candidate)

    for cmd in ("musescore4", "musescore", "mscore", "MuseScore4"):
        found = shutil.which(cmd)
        if found:
            return found

    windows_candidates = (
        Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"),
        Path(r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe"),
    )
    for path in windows_candidates:
        if path.is_file():
            return str(path)
    return None


def convert_with_musescore_cli(musicxml_path: Path, output_format: str = "mscz") -> Path | None:
    """Convert MusicXML to .mscz or .pdf using local MuseScore CLI if installed."""
    fmt = output_format.strip().lower().lstrip(".")
    if fmt not in {"mscz", "pdf", "mid", "mxl"}:
        raise SecurityValidationError(f"Unsupported MuseScore output format: {output_format}")

    ms_bin = find_musescore_binary()
    if not ms_bin:
        return None

    target_path = musicxml_path.with_suffix(f".{fmt}")
    run_safe_command([ms_bin, "-o", str(target_path), "--", str(musicxml_path)], timeout=120)
    return target_path if target_path.is_file() else None
