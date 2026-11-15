"""Audio-to-MIDI bass transcription using Spotify Basic Pitch, librosa, or built-in YIN."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

DEFAULT_BASS_MIN_MIDI = 28  # E1 (41.20 Hz)
DEFAULT_BASS_MAX_MIDI = 67  # G4 (392.00 Hz)
FIVE_STRING_MIN_MIDI = 23   # B0 (30.87 Hz)

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class BassNote:
    """Single transcribed bass note event."""

    pitch_midi: int
    note_name: str
    onset_sec: float
    duration_sec: float
    velocity: int = 96
    pitch_bend: int = 0


def has_basic_pitch() -> bool:
    """Return True if spotify/basic-pitch is installed."""
    return importlib.util.find_spec("basic_pitch") is not None


def has_librosa() -> bool:
    """Return True if librosa is installed."""
    return importlib.util.find_spec("librosa") is not None


def has_beat_this() -> bool:
    """Return True if CPJKU/beat_this is installed."""
    return importlib.util.find_spec("beat_this") is not None


def detect_audio_downbeats(audio_wav: Path) -> tuple[list[float], list[float]]:
    """Optionally detect (beats, downbeats) timestamps via CPJKU/beat_this when installed."""
    if not has_beat_this() or not audio_wav.is_file():
        return [], []
    try:
        from beat_this.inference import File2Beats

        beats, downbeats = File2Beats()(str(audio_wav))
        return [float(x) for x in beats], [float(x) for x in downbeats]
    except Exception:
        return [], []


def estimate_track_start_sec(audio_wav: Path | None) -> float:
    """Estimate first audible musical sound / beat in the full mix (clamped to [0, 15]s)."""
    if audio_wav is None or not Path(audio_wav).is_file():
        return 0.0
    try:
        audio, sr = _load_mono_audio(Path(audio_wav), target_sr=22050)
    except Exception:
        return 0.0

    win, hop = 1024, 512
    n_frames = (len(audio) - win) // hop + 1
    if n_frames < 3:
        return 0.0

    rms = np.array(
        [float(np.sqrt(np.mean(audio[i * hop : i * hop + win] ** 2))) for i in range(n_frames)],
        dtype=np.float32,
    )
    peak = float(np.max(rms))
    if peak < 1e-4:
        return 0.0

    thresh = max(peak * 0.06, 1e-4)
    first_rms_sec = 0.0
    for i in range(n_frames - 2):
        if rms[i] >= thresh and rms[i + 1] >= thresh and rms[i + 2] >= thresh:
            first_rms_sec = (i * hop) / float(sr)
            break

    start_sec = first_rms_sec
    if has_beat_this():
        beats, _ = detect_audio_downbeats(Path(audio_wav))
        if beats:
            start_sec = min(float(beats[0]), first_rms_sec)
    return round(max(0.0, min(15.0, float(start_sec))), 4)


def midi_to_note_name(midi_num: int) -> str:
    """Convert an integer MIDI pitch to scientific pitch notation (e.g. 28 -> 'E1')."""
    octave = (midi_num // 12) - 1
    return f"{_NOTE_NAMES[midi_num % 12]}{octave}"


def midi_to_hz(midi_num: float) -> float:
    """Convert MIDI pitch number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_num - 69.0) / 12.0))


def hz_to_midi(freq_hz: float) -> float:
    """Convert frequency in Hz to fractional MIDI pitch."""
    return 69.0 + 12.0 * float(np.log2(max(freq_hz, 1e-6) / 440.0))


def _load_mono_audio(wav_path: Path, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        audio = data.astype(np.float32) / max(float(np.iinfo(data.dtype).max), 1.0)
    else:
        audio = data.astype(np.float32)

    if sr != target_sr and sr > 0:
        g = gcd(target_sr, int(sr))
        audio = resample_poly(audio, target_sr // g, int(sr) // g).astype(np.float32)
        sr = target_sr
    return audio, int(sr)


def _yin_pitch_frame(frame: np.ndarray, sr: int, tau_min: int, tau_max: int, threshold: float = 0.18) -> float | None:
    """Estimate fundamental frequency (Hz) for a single audio frame using YIN CMNDF."""
    half = len(frame) // 2
    if half <= tau_max:
        return None
    w = half
    x = frame[:w].astype(np.float64)
    y = frame[: 2 * w].astype(np.float64)

    # Compute d(tau) = sum_{j=0}^{w-1} (y[j] - y[j+tau])^2 via prefix sums + FFT cross-correlation.
    n_fft = 1 << int(np.ceil(np.log2(2 * w + w)))
    x_fft = np.fft.rfft(x[::-1], n=n_fft)
    y_fft = np.fft.rfft(y, n=n_fft)
    conv = np.fft.irfft(x_fft * y_fft, n=n_fft)
    acf = conv[w - 1 : w - 1 + w]

    sq = y * y
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    energy_x = csum[w]
    energy_y_shifted = csum[w : 2 * w] - csum[:w]
    d = np.maximum(energy_x + energy_y_shifted - 2.0 * acf, 0.0)

    cmndf = np.ones(w, dtype=np.float64)
    running_sum = np.cumsum(d[1:])
    nonzero = running_sum > 1e-12
    cmndf[1:][nonzero] = d[1:][nonzero] * np.arange(1, w)[nonzero] / running_sum[nonzero]

    tau = tau_min
    while tau < tau_max:
        if cmndf[tau] < threshold:
            while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            break
        tau += 1
    else:
        best_tau = int(np.argmin(cmndf[tau_min:tau_max])) + tau_min
        if cmndf[best_tau] >= 0.32:
            return None
        tau = best_tau

    # Guard against 2nd-harmonic octave lock on higher candidates (f > 165 Hz).
    if tau > 0 and (float(sr) / float(tau)) > 165.0 and (2 * tau) < (tau_max - 1):
        sub_start = max(tau_min, 2 * tau - 3)
        sub_end = min(tau_max, 2 * tau + 4)
        sub_window = cmndf[sub_start:sub_end]
        if len(sub_window) > 0 and float(np.min(sub_window)) < 0.30:
            tau = sub_start + int(np.argmin(sub_window))

    # Parabolic interpolation around tau for sub-sample frequency precision.
    if 1 <= tau < tau_max - 1:
        s0, s1, s2 = cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]
        denom = 2.0 * (2.0 * s1 - s2 - s0)
        if abs(denom) > 1e-12:
            tau_refined = tau + (s0 - s2) / denom
        else:
            tau_refined = float(tau)
    else:
        tau_refined = float(tau)

    return float(sr) / max(tau_refined, 1.0)


def _suppress_semitone_wobble(
    smoothed: np.ndarray,
    min_midi: int,
    max_midi: int,
    onset_frames: set[int] | None = None,
) -> np.ndarray:
    """Collapse short +-1 semitone attack/release glitches into adjacent steady pitches."""
    runs: list[list[int]] = []
    n = len(smoothed)
    idx = 0
    while idx < n:
        p = int(smoothed[idx])
        if min_midi <= p <= max_midi:
            end = idx + 1
            while end < n and int(smoothed[end]) == p:
                end += 1
            runs.append([idx, end, p])
            idx = end
        else:
            idx += 1

    if len(runs) < 2:
        return smoothed

    # Pass 1: snap <=7-frame (~81ms) semitone glitches attached directly to steady notes (>=8 frames).
    for r_i, (start, end, pitch) in enumerate(runs):
        length = end - start
        if length > 7:
            continue
        target: int | None = None
        if r_i > 0:
            l_start, l_end, l_pitch = runs[r_i - 1]
            no_onset_split = onset_frames is None or start not in onset_frames
            if l_end == start and (l_end - l_start) >= 8 and abs(pitch - l_pitch) == 1 and no_onset_split:
                target = l_pitch
        if target is None and r_i + 1 < len(runs):
            n_start, n_end, n_pitch = runs[r_i + 1]
            no_onset_split = onset_frames is None or n_start not in onset_frames
            if end == n_start and (n_end - n_start) >= 8 and abs(pitch - n_pitch) == 1:
                nearby_same = sum(
                    1 for j in range(max(0, r_i - 2), min(len(runs), r_i + 4)) if j != r_i and runs[j][2] == n_pitch
                )
                if no_onset_split or nearby_same >= 2:
                    target = n_pitch
        if target is not None:
            smoothed[start:end] = target
            runs[r_i][2] = target

    # Pass 2: remove <=6-frame (~70ms) semitone dips sandwiched between repeated identical pitches.
    for r_i in range(1, len(runs) - 1):
        start, end, pitch = runs[r_i]
        if (end - start) > 6:
            continue
        _, l_end, l_pitch = runs[r_i - 1]
        n_start, _, n_pitch = runs[r_i + 1]
        if l_pitch == n_pitch and abs(pitch - l_pitch) == 1 and (start - l_end) <= 2 and (n_start - end) <= 2:
            smoothed[start:end] = l_pitch
            runs[r_i][2] = l_pitch

    return smoothed


def _segments_from_frame_pitches(
    midi_frames: np.ndarray,
    rms_frames: np.ndarray,
    hop_sec: float,
    min_midi: int,
    max_midi: int,
    min_note_len_sec: float,
    onset_frames: set[int] | None = None,
) -> list[BassNote]:
    """Group quantized per-frame MIDI values into monophonic BassNote events."""
    valid = (midi_frames >= min_midi) & (midi_frames <= max_midi)
    if not np.any(valid):
        return []

    # Median-smooth adjacent voiced frames without bleeding pitches across rests.
    smoothed = midi_frames.copy()
    n = len(midi_frames)
    for idx in range(1, n - 1):
        if valid[idx]:
            window = [int(midi_frames[j]) for j in (idx - 1, idx, idx + 1) if valid[j]]
            if len(window) >= 2:
                window.sort()
                smoothed[idx] = window[len(window) // 2]

    smoothed = _suppress_semitone_wobble(smoothed, min_midi, max_midi, onset_frames)

    peak_rms = float(np.max(rms_frames)) if len(rms_frames) else 1.0
    notes: list[BassNote] = []

    current_pitch = -1
    start_idx = 0
    seg_rms: list[float] = []

    def _flush(end_idx: int) -> None:
        if current_pitch < min_midi or current_pitch > max_midi:
            return
        dur = (end_idx - start_idx) * hop_sec
        if dur < min_note_len_sec:
            return
        mean_rms = float(np.mean(seg_rms)) if seg_rms else 0.5 * peak_rms
        vel = int(np.clip(45 + 75 * (mean_rms / max(peak_rms, 1e-6)), 40, 120))
        notes.append(
            BassNote(
                pitch_midi=int(current_pitch),
                note_name=midi_to_note_name(int(current_pitch)),
                onset_sec=round(start_idx * hop_sec, 4),
                duration_sec=round(dur, 4),
                velocity=vel,
            )
        )

    for idx in range(len(smoothed)):
        p = int(smoothed[idx])
        r = float(rms_frames[idx])
        if p < min_midi or p > max_midi:
            _flush(idx)
            current_pitch = -1
            seg_rms = []
            continue

        # Detect re-articulated repeated note via onset event or RMS dip valley.
        if onset_frames is not None:
            reattack = (
                p == current_pitch
                and len(seg_rms) >= 4
                and (
                    idx in onset_frames
                    or (seg_rms[-1] < 0.55 * max(seg_rms[:-1]) and r > 1.3 * seg_rms[-1])
                )
            )
        else:
            reattack = (
                p == current_pitch
                and len(seg_rms) >= 3
                and seg_rms[-1] < 0.35 * max(seg_rms[:-1])
                and r > 1.6 * seg_rms[-1]
            )
        if p != current_pitch or reattack:
            _flush(idx)
            current_pitch = p
            start_idx = idx
            seg_rms = [r]
        else:
            seg_rms.append(r)

    _flush(len(smoothed))
    return notes


def _transcribe_builtin_yin(
    bass_wav: Path,
    min_midi: int,
    max_midi: int,
    min_note_len_sec: float,
) -> list[BassNote]:
    audio, sr = _load_mono_audio(bass_wav, target_sr=22050)
    if len(audio) < 1600:
        return []

    frame_len = 1600
    hop = 256
    f_min = max(25.0, midi_to_hz(min_midi) * 0.9)
    f_max = min(500.0, midi_to_hz(max_midi) * 1.1)
    tau_min = max(16, int(sr / f_max))
    tau_max = min(frame_len // 2 - 2, int(sr / f_min))

    n_frames = max(0, (len(audio) - frame_len) // hop + 1)
    if n_frames == 0:
        return []

    rms_frames = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = audio[i * hop : i * hop + frame_len]
        rms_frames[i] = float(np.sqrt(np.mean(seg * seg)))

    peak_rms = float(np.max(rms_frames)) if len(rms_frames) else 0.0
    if peak_rms < 1e-5:
        return []

    active_rms = rms_frames[rms_frames > peak_rms * 0.04]
    if len(active_rms) > 0:
        p25 = float(np.percentile(active_rms, 25))
        p85 = float(np.percentile(active_rms, 85))
        if p85 > 2.2 * max(p25, 1e-8):
            rms_gate = max(peak_rms * 0.17, p25 * 1.25)
        else:
            rms_gate = peak_rms * 0.05
    else:
        rms_gate = peak_rms * 0.05

    midi_frames = np.full(n_frames, -1, dtype=np.int32)
    for i in range(n_frames):
        if rms_frames[i] < rms_gate:
            continue
        seg = audio[i * hop : i * hop + frame_len]
        hz = _yin_pitch_frame(seg, sr, tau_min, tau_max)
        if hz is not None and f_min <= hz <= f_max:
            midi_frames[i] = int(round(hz_to_midi(hz)))

    notes = _segments_from_frame_pitches(
        midi_frames=midi_frames,
        rms_frames=rms_frames,
        hop_sec=hop / float(sr),
        min_midi=min_midi,
        max_midi=max_midi,
        min_note_len_sec=min_note_len_sec,
    )
    return _clean_transcribed_notes(notes)


def _transcribe_basic_pitch(
    bass_wav: Path,
    min_midi: int,
    max_midi: int,
    min_note_len_sec: float,
) -> list[BassNote]:
    from basic_pitch.inference import predict

    _, _, note_events = predict(
        str(bass_wav),
        minimum_frequency=float(midi_to_hz(min_midi) * 0.9),
        maximum_frequency=float(midi_to_hz(max_midi) * 1.1),
    )
    notes: list[BassNote] = []
    for start_s, end_s, pitch, amp, pitch_bends in note_events:
        pitch_int = int(round(pitch))
        dur = float(end_s - start_s)
        if not (min_midi <= pitch_int <= max_midi) or dur < min_note_len_sec:
            continue
        bend = int(round(float(np.mean(pitch_bends)))) if pitch_bends else 0
        vel = int(np.clip(round(float(amp) * 127), 35, 127))
        notes.append(
            BassNote(
                pitch_midi=pitch_int,
                note_name=midi_to_note_name(pitch_int),
                onset_sec=round(float(start_s), 4),
                duration_sec=round(dur, 4),
                velocity=vel,
                pitch_bend=bend,
            )
        )
    return _clean_transcribed_notes(_enforce_monophonic_bass(notes))


def _transcribe_librosa_pyin(
    bass_wav: Path,
    min_midi: int,
    max_midi: int,
    min_note_len_sec: float,
    bpm_hint: float | None = None,
) -> list[BassNote]:
    import librosa

    audio, sr = _load_mono_audio(bass_wav, target_sr=22050)
    hop = 256
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=float(midi_to_hz(min_midi) * 0.92),
        fmax=float(midi_to_hz(max_midi) * 1.08),
        sr=sr,
        hop_length=hop,
    )
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=hop)[0]
    fast_tempo = bpm_hint is not None and bpm_hint >= 145.0
    onset_frames = {
        int(x)
        for x in librosa.onset.onset_detect(
            y=audio,
            sr=sr,
            hop_length=hop,
            backtrack=False,
            delta=0.045 if fast_tempo else 0.03,
            wait=6 if fast_tempo else 4,
        )
    }
    n = min(len(f0), len(rms))
    midi_frames = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        if voiced_flag[i] and np.isfinite(f0[i]):
            midi_frames[i] = int(round(hz_to_midi(float(f0[i]))))

    # Compensate for 2048-sample pYIN window lag (1-3 frames) behind sharp pluck transients.
    for i in range(3, n):
        if midi_frames[i] >= min_midi and midi_frames[i - 1] < min_midi:
            for back in (3, 2, 1):
                if (i - back) in onset_frames and rms[i - back] >= 0.25 * rms[i]:
                    midi_frames[i - back : i] = midi_frames[i]
                    break

    notes = _segments_from_frame_pitches(
        midi_frames=midi_frames,
        rms_frames=rms[:n],
        hop_sec=hop / float(sr),
        min_midi=min_midi,
        max_midi=max_midi,
        min_note_len_sec=min_note_len_sec,
        onset_frames=onset_frames,
    )
    return _clean_transcribed_notes(notes, min_velocity=62)


def _clean_transcribed_notes(notes: list[BassNote], min_velocity: int = 66) -> list[BassNote]:
    """Filter low-energy bleed/harmonic ghosts and merge micro-split same-pitch plucks."""
    if not notes:
        return []

    max_vel = max(n.velocity for n in notes)
    filtered: list[BassNote] = []
    for n in notes:
        if max_vel >= 75 and n.velocity < min_velocity:
            continue
        if n.pitch_midi >= 50 and (n.velocity < 78 or n.duration_sec < 0.11):
            continue
        filtered.append(n)

    filtered.sort(key=lambda item: item.onset_sec)
    # Snap short <=100ms +-1 semitone attack/release glitches to surrounding repeated pitches.
    for idx in range(len(filtered)):
        cur = filtered[idx]
        if cur.duration_sec > 0.10:
            continue
        neighbors = [filtered[j].pitch_midi for j in range(max(0, idx - 2), min(len(filtered), idx + 3)) if j != idx]
        for target_p in set(neighbors):
            if abs(cur.pitch_midi - target_p) == 1 and neighbors.count(target_p) >= 2:
                prev_ok = (
                    idx > 0
                    and filtered[idx - 1].pitch_midi == target_p
                    and (cur.onset_sec - (filtered[idx - 1].onset_sec + filtered[idx - 1].duration_sec)) <= 0.025
                )
                next_ok = (
                    idx + 1 < len(filtered)
                    and filtered[idx + 1].pitch_midi == target_p
                    and (filtered[idx + 1].onset_sec - (cur.onset_sec + cur.duration_sec)) <= 0.025
                )
                if prev_ok or next_ok:
                    filtered[idx] = BassNote(
                        pitch_midi=target_p,
                        note_name=midi_to_note_name(target_p),
                        onset_sec=cur.onset_sec,
                        duration_sec=cur.duration_sec,
                        velocity=cur.velocity,
                        pitch_bend=cur.pitch_bend,
                    )
                    break

    merged: list[BassNote] = []
    for cur in filtered:
        if merged:
            prev = merged[-1]
            gap = cur.onset_sec - (prev.onset_sec + prev.duration_sec)
            same_pitch = cur.pitch_midi == prev.pitch_midi
            is_micro_split = same_pitch and gap <= 0.045 and (cur.onset_sec - prev.onset_sec) < 0.09
            is_glitch_split = (
                same_pitch
                and gap <= 0.015
                and min(prev.duration_sec, cur.duration_sec) <= 0.095
                and (cur.onset_sec + cur.duration_sec - prev.onset_sec) <= 0.24
            )
            if is_micro_split or is_glitch_split:
                new_end = max(prev.onset_sec + prev.duration_sec, cur.onset_sec + cur.duration_sec)
                merged[-1] = BassNote(
                    pitch_midi=prev.pitch_midi,
                    note_name=prev.note_name,
                    onset_sec=prev.onset_sec,
                    duration_sec=round(max(0.03, new_end - prev.onset_sec), 4),
                    velocity=max(prev.velocity, cur.velocity),
                    pitch_bend=prev.pitch_bend,
                )
                continue
        merged.append(cur)
    return merged


def _enforce_monophonic_bass(notes: list[BassNote]) -> list[BassNote]:
    """Sort notes by onset and trim overlaps so a bass line stays clean and playable."""
    if not notes:
        return []
    ordered = sorted(notes, key=lambda n: (n.onset_sec, -n.velocity))
    cleaned: list[BassNote] = []
    for note in ordered:
        if cleaned:
            prev = cleaned[-1]
            # Merge/drop simultaneous double-trigger within 25 ms by keeping louder note.
            if abs(note.onset_sec - prev.onset_sec) < 0.025:
                if note.velocity > prev.velocity:
                    cleaned[-1] = note
                continue
            prev_end = prev.onset_sec + prev.duration_sec
            if prev_end > note.onset_sec:
                trimmed_dur = round(max(0.04, note.onset_sec - prev.onset_sec), 4)
                cleaned[-1] = BassNote(
                    pitch_midi=prev.pitch_midi,
                    note_name=prev.note_name,
                    onset_sec=prev.onset_sec,
                    duration_sec=trimmed_dur,
                    velocity=prev.velocity,
                    pitch_bend=prev.pitch_bend,
                )
        cleaned.append(note)
    return cleaned


def estimate_bpm(notes: list[BassNote], fallback_bpm: float = 120.0) -> float:
    """Estimate a practical tempo in [60, 180] BPM via IOI clustering and 8th-grid phase locking."""
    if len(notes) < 3:
        return float(fallback_bpm)
    onsets = np.array([n.onset_sec for n in notes], dtype=np.float64)
    iois = np.diff(onsets)
    iois = iois[(iois >= 0.15) & (iois <= 2.0)]
    if len(iois) == 0:
        return float(fallback_bpm)

    median_ioi = float(np.median(iois))
    cluster = iois[np.abs(iois - median_ioi) <= 0.18 * median_ioi]
    cluster_ioi = float(np.median(cluster)) if len(cluster) > 0 else median_ioi

    base_bpm = 60.0 / max(cluster_ioi, 1e-3)
    while base_bpm < 70.0:
        base_bpm *= 2.0
    while base_bpm > 175.0:
        base_bpm *= 0.5

    sample_notes = notes[:200]
    t0 = sample_notes[0].onset_sec
    rel_onsets = np.array([n.onset_sec - t0 for n in sample_notes], dtype=np.float64)
    vels = np.array([float(n.velocity) for n in sample_notes], dtype=np.float64)
    sum_vel = max(float(np.sum(vels)), 1.0)

    best_bpm = base_bpm
    best_score = -float("inf")
    low_cand = max(60.0, base_bpm - 2.0)
    high_cand = min(180.0, base_bpm + 2.0)
    for cand_bpm in np.linspace(low_cand, high_cand, 81):
        step_sec = 30.0 / max(float(cand_bpm), 20.0)
        phase = 2.0 * np.pi * rel_onsets / step_sec
        score = float(np.sum(vels * np.cos(phase))) - 0.15 * abs(float(cand_bpm) - base_bpm) * sum_vel
        if score > best_score:
            best_score = score
            best_bpm = float(cand_bpm)

    return round(best_bpm, 1)


def transcribe_bass_notes(
    bass_wav: Path,
    min_midi: int = DEFAULT_BASS_MIN_MIDI,
    max_midi: int = DEFAULT_BASS_MAX_MIDI,
    min_note_len_ms: int = 60,
    bpm_hint: float | None = None,
) -> dict:
    """Transcribe a bass WAV file into discrete bass notes and tempo estimate."""
    min_note_len_sec = max(0.03, min_note_len_ms / 1000.0)

    if has_basic_pitch():
        notes = _transcribe_basic_pitch(bass_wav, min_midi, max_midi, min_note_len_sec)
        engine = "spotify:basic-pitch"
    elif has_librosa():
        notes = _transcribe_librosa_pyin(
            bass_wav, min_midi, max_midi, min_note_len_sec, bpm_hint=bpm_hint
        )
        engine = "librosa:pyin"
    else:
        notes = _transcribe_builtin_yin(bass_wav, min_midi, max_midi, min_note_len_sec)
        engine = "builtin:yin-autocorr"

    bpm = float(bpm_hint) if bpm_hint and bpm_hint > 0 else estimate_bpm(notes)
    return {
        "notes": notes,
        "engine": engine,
        "estimated_bpm": bpm,
    }
