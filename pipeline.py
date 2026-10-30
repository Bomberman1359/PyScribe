# main transcriber pipeline code

import collections, collections.abc
collections.MutableSequence = collections.abc.MutableSequence
import numpy as np
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'):   np.int = int
if not hasattr(np, 'bool'):  np.bool = bool

import os, sys, subprocess, warnings, copy, bisect
warnings.filterwarnings("ignore")

import scipy.signal, scipy.signal.windows
if not hasattr(scipy.signal, 'hann'):
    scipy.signal.hann = scipy.signal.windows.hann

import librosa
import soundfile as sf
import pretty_midi
import music21 as m21

try:
    import madmom
    HAS_MADMOM = True
except Exception:
    HAS_MADMOM = False


# settings
SR  = 16000
HOP = 160
DEMUCS_MODEL = "htdemucs"
STEM_ROOT    = "separated"
OUT_ROOT     = "output"
FORCE_REBUILD = False

BEAT_REPAIR_VERSION = 2
GRID_FIT_TOL      = 0.20
GRID_FIT_COVERAGE = 0.85
BEAT_GAP_TOL    = 0.18
BEAT_LOCAL_WIN  = 9
BEAT_STEADY_MAX = 0.12
BEAT_STEADY_RMS = 0.04

CREPE_MODEL     = "full"
CREPE_CHUNK_SEC = 30
CREPE_FMIN, CREPE_FMAX = 60, 1000

V_DEPTH_RATIO   = 0.45
V_MIN_FRAMES    = 8
V_CONF_THRESH   = 0.50
V_PITCH_TOL     = 0.60
V_BRIDGE_FRAMES = 10
V_MIDI_LO, V_MIDI_HI = 48, 84

EXT_HOLD_FRAC    = 0.25
EXT_MAX_FILL_SEC = 1.0

CH_HOP = 512
CH_BASS_REGISTER = 3
CH_ROOT_WEIGHT, CH_THIRD_WEIGHT, CH_FIFTH_WEIGHT = 1.6, 1.0, 1.0
CH_BASS_VETO_MULT, CH_BASS_PENALTY_MULT = 1.5, 0.5
CH_LOW_CONF_MARGIN = 0.55

U_PER_BEAT  = 4
U_PER_BAR   = 16
BASS_OCTAVE = 3
CHORD_SNAP_UNITS  = 4
LEGATO_FILL_UNITS = 2
MELODY_COARSE     = 2
RUN_KEEP_16TH     = True
RUN_MIN_LEN       = 2
MERGE_GAP_UNITS   = 0
FRAGMENT_UNITS    = 1
FRAGMENT_MERGE_MAX = 2
DD_FIX       = {7: 6, 14: 12}
CLEAN_DURS   = {1, 2, 3, 4, 6, 8, 12, 16}
GS_VERSION   = 15
MEL_VERSION  = 5

SIL_FLOOR_FRAC = 0.15
SIL_GATE_BEATS = 2.0
ACC_SMOOTH_SEC = 3.0

TREBLE_LOW_MEDIAN = 55
TREBLE_TINY_LOW   = 50
TREBLE_PHRASE_GAP = 8
BASS_DROP_FLOOR   = 24
CHROMA_BLIP_MAX   = 2
CHD_FIX_MAX_BEATS = 2.0

ENABLE_REST_FILL = True
VGAP_GATE_BEATS = 2.0
VPATCH_CONF     = 0.50
VPATCH_MIN_BEATS = 4.0

VOCAL_FLOOR_FRAC  = 0.15
VOCAL_ACTIVE_MIN  = 0.35
VOCAL_OTHER_SHARE = 0.20

ZIGZAG_MIN_NOTES = 7
ZIGZAG_FRAC      = 0.6
ZIGZAG_LEAP      = 3
FILL_GATE_BEATS  = 4.0
FILL_MIN_NOTES   = 2

SEED_MELODY_JITTER = 0.35
MEDIUM_MENU = ["pulse", "asc_ninth", "asc_twelfth"]
BUSY_MENU   = ["climb", "updown", "pump"]
CALM_MENU   = ["block", "inv1", "inv2", "open5", "oct158"]


# vocal note helpers
def hz_to_midi(f):
    f = np.asarray(f, float); out = np.full_like(f, np.nan); m = f > 0
    out[m] = 69 + 12 * np.log2(f[m] / 440.0); return out

def _median_smooth(x, voiced, win=17):
    out = x.copy(); h = win // 2; n = len(x)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        seg = x[lo:hi][voiced[lo:hi]]
        if len(seg): out[i] = np.median(seg)
    return out

def _smooth(x, win=5):
    if win < 2: return x.copy()
    k = np.ones(win) / win
    return np.convolve(x, k, mode='same')

def rearticulations_from_rms(rms, voiced, depth_ratio=0.55, look=20, min_sep=8):
    sm = _smooth(np.asarray(rms, float), 3); n = len(sm)
    art = np.zeros(n, bool); last = -min_sep
    for i in range(1, n - 1):
        if not voiced[i]: continue
        if sm[i] <= sm[i - 1] and sm[i] < sm[i + 1]:
            L = max(0, i - look); R = min(n, i + look)
            left = sm[L:i].max() if i > L else sm[i]
            right = sm[i + 1:R].max() if R > i + 1 else sm[i]
            ref = min(left, right)
            if ref > 1e-9 and sm[i] <= depth_ratio * ref and (i - last) >= min_sep:
                art[i] = True; last = i
    return art

def extract_notes_fused_v2(times, f0, conf, rms,
                           conf_thresh=0.5, pitch_tol=0.6, min_frames=4,
                           bridge_frames=10, depth_ratio=0.55, smooth_win=17,
                           min_plateau=6, rms_floor_ratio=0.15):
    midi = hz_to_midi(f0)
    voiced = (conf >= conf_thresh) & np.isfinite(midi)
    base = np.where(voiced, midi, np.nan)
    base = np.where(np.isfinite(base), base, np.where(np.isfinite(midi), midi, 0.0))
    sm = _median_smooth(base, voiced, smooth_win)
    art = rearticulations_from_rms(rms, voiced, depth_ratio)
    rms = np.asarray(rms, float)
    vr = rms[voiced]
    rms_floor = (np.median(vr) * rms_floor_ratio) if len(vr) else 0.0

    def look_dev(i, ref):
        seg = sm[i:i + min_plateau]; seg = seg[np.isfinite(seg)]
        return abs(np.median(seg) - ref) if len(seg) else 0.0

    notes = []; i = 0; n = len(times)
    while i < n:
        if not voiced[i]: i += 1; continue
        start = i; ref = sm[i]; i += 1
        while i < n:
            if not voiced[i]:
                j = i
                while j < n and not voiced[j]: j += 1
                gap = j - i; attack = art[i:j + 1].any()
                if gap <= bridge_frames and j < n and abs(sm[j] - ref) <= pitch_tol and not attack:
                    i = j; continue
                break
            if art[i]: break
            if abs(sm[i] - ref) > pitch_tol:
                if look_dev(i, ref) > pitch_tol:
                    break
                i += 1; continue
            ref = np.median(sm[start:i + 1]); i += 1
        seg_rms = np.median(rms[start:i]) if i > start else 0.0
        if (i - start) >= min_frames and seg_rms >= rms_floor:
            seg = sm[start:i]; notes.append((round(float(times[start]), 3),
                round(float(times[min(i, n - 1)]), 3), int(round(np.median(seg)))))
        i = max(i, start + 1)
    return notes

def fix_octave_outliers(notes, window=5, jump=11):
    if len(notes) < 3: return notes
    p = np.array([x[2] for x in notes], float)
    center = np.array([np.median(p[max(0, k - window):min(len(p), k + window + 1)])
                       for k in range(len(p))])
    out = []
    for k, (s, e, pk) in enumerate(notes):
        med = center[k]; cand = pk
        if abs(pk - med) >= jump:
            while cand - med > 6: cand -= 12
            while med - cand > 6: cand += 12
        out.append((s, e, int(cand)))
    return out


def fix_intro_octaves(notes, intro_sec=8.0, jump=10, min_reliable=8):
    if len(notes) < min_reliable: return notes
    pitches = np.array([p for *_, p in notes], float); starts = np.array([s for s, _, _ in notes], float)
    body = pitches[starts >= intro_sec]
    center = float(np.median(body)) if len(body) >= min_reliable else float(np.median(pitches))
    out = []
    for (s, e, p) in notes:
        if s < intro_sec and abs(p - center) >= jump:
            while p - center > 6: p -= 12
            while center - p > 6: p += 12
        out.append((s, e, int(p)))
    return out


# chord helpers
def note_names():
    return ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def generate_templates():
    names = note_names()
    maj_base = np.zeros(12)
    maj_base[0] = CH_ROOT_WEIGHT; maj_base[4] = CH_THIRD_WEIGHT; maj_base[7] = CH_FIFTH_WEIGHT
    maj_base /= np.linalg.norm(maj_base)
    min_base = np.zeros(12)
    min_base[0] = CH_ROOT_WEIGHT; min_base[3] = CH_THIRD_WEIGHT; min_base[7] = CH_FIFTH_WEIGHT
    min_base /= np.linalg.norm(min_base)
    templates = {}
    for i in range(12):
        root_midi = 12 * (CH_BASS_REGISTER + 1) + i
        templates[f"{names[i]} Major"] = {'vector': np.roll(maj_base, i),
            'midi': [root_midi, root_midi + 4, root_midi + 7], 'root': i}
        templates[f"{names[i]} Minor"] = {'vector': np.roll(min_base, i),
            'midi': [root_midi, root_midi + 3, root_midi + 7], 'root': i}
    return templates

def match_chord(chroma_other, chroma_bass, start, end, templates, names, sr, hop):
    f0 = librosa.time_to_frames(start, sr=sr, hop_length=hop)
    f1 = librosa.time_to_frames(end, sr=sr, hop_length=hop)
    f1 = max(f1, f0 + 1)
    f1 = min(f1, chroma_other.shape[1])
    slice_other = chroma_other[:, f0:f1]
    slice_bass = chroma_bass[:, f0:f1]
    if slice_other.shape[1] == 0:
        return None
    vec_other = np.median(slice_other, axis=1)
    norm_other = np.linalg.norm(vec_other)
    if norm_other > 0:
        vec_other = vec_other / norm_other
    vec_bass = np.median(slice_bass, axis=1)
    norm_bass = np.linalg.norm(vec_bass)
    if norm_bass > 0:
        vec_bass = vec_bass / norm_bass
    bass_note = int(np.argmax(vec_bass))
    bass_strength = vec_bass[bass_note]
    scores = []
    for name in names:
        template = templates[name]
        base_score = np.dot(template['vector'], vec_other)
        if template['root'] == bass_note:
            score = base_score * (1.0 + CH_BASS_VETO_MULT * bass_strength)
        else:
            score = base_score * (1.0 - CH_BASS_PENALTY_MULT * bass_strength)
        scores.append(score)
    scores = np.array(scores)
    order = np.argsort(scores)[::-1]
    best_score = scores[order[0]]
    second_score = scores[order[1]]
    margin = best_score / (best_score + second_score + 1e-9)
    return {'name': names[order[0]], 'margin': margin}

def snap_to_grid(t, grid_unit):
    return round(t / grid_unit) * grid_unit

def fix_nondiatonic_blocks(merged, templates, beat_dur):
    if len(merged) < 3: return merged, 0
    pcset = lambda name: {p % 12 for p in templates[name]['midi']}
    best_set, best_cov = set(range(12)), -1.0
    for tonic in range(12):
        dia = {(tonic + i) % 12 for i in (0, 2, 4, 5, 7, 9, 11)}
        cov = sum((c['end'] - c['start']) for c in merged if pcset(c['name']) <= dia)
        if cov > best_cov: best_cov, best_set = cov, dia
    med_margin = float(np.median([m for c in merged for m in c['margins']]))
    fixed, res = 0, []
    i = 0
    while i < len(merged):
        c = merged[i]
        beats = (c['end'] - c['start']) / beat_dur
        weak = float(np.mean(c['margins'])) < med_margin
        offkey = not (pcset(c['name']) <= best_set)
        if offkey and weak and beats <= CHD_FIX_MAX_BEATS + 1e-6:
            prv = res[-1] if res else None
            nxt = merged[i + 1] if i + 1 < len(merged) else None
            if prv is not None and (not (pcset(prv['name']) <= best_set)
                                    or abs(prv['end'] - c['start']) > 1e-3):
                prv = None
            if nxt is not None and (not (pcset(nxt['name']) <= best_set)
                                    or abs(c['end'] - nxt['start']) > 1e-3):
                nxt = None
            take = None
            if prv is not None and nxt is not None:
                take = prv if (prv['end'] - prv['start']) >= (nxt['end'] - nxt['start']) else nxt
            else:
                take = prv or nxt
            if take is prv and prv is not None:
                prv['end'] = c['end']; prv['margins'] += c['margins']
                fixed += 1; i += 1; continue
            if take is nxt and nxt is not None:
                nxt['start'] = c['start']; nxt['margins'] = c['margins'] + nxt['margins']
                fixed += 1; i += 1; continue
        res.append(c); i += 1
    out = []
    for c in res:
        if out and out[-1]['name'] == c['name'] and abs(out[-1]['end'] - c['start']) < 1e-3:
            out[-1]['end'] = c['end']; out[-1]['margins'] += c['margins']
        else:
            out.append(c)
    return out, fixed


# stage 9 grand staff
def q_to_u(ql): return int(round(float(ql) * U_PER_BEAT))
def u_to_q(u):  return u * (1.0 / U_PER_BEAT)
def snap(u, grid): return int(round(u / grid)) * grid

def clean_melody_rhythm(events, coarse=MELODY_COARSE):
    if not events: return events
    on  = [e['on'] for e in events]
    end = [e['on'] + e['dur'] for e in events]
    n = len(events)
    fast = [False] * n
    if RUN_KEEP_16TH:
        dur = [end[k] - on[k] for k in range(n)]
        linked = [k < n - 1 and dur[k] == 1 and dur[k + 1] == 1
                  and on[k + 1] - on[k] == 1 for k in range(n)]
        k = 0
        while k < n:
            j = k
            while j < n - 1 and linked[j]: j += 1
            if (j - k + 1) >= RUN_MIN_LEN:
                for i in range(k, j + 1): fast[i] = True
            k = j + 1
    out = []
    for i, e in enumerate(events):
        grid = 1 if fast[i] else coarse
        no = int(round(on[i] / grid)) * grid
        if out and no <= out[-1]['on']:
            no = out[-1]['on'] + (1 if fast[i] else coarse)
        ne = int(round(end[i] / grid)) * grid
        if ne <= no: ne = no + grid
        out.append({'on': no, 'dur': ne - no, 'kind': 'note', 'payload': e['payload']})
    return out

def fold_octave_outliers_events(events, window=5, jump=11):
    if len(events) < 3: return events
    p = np.array([e['payload'] for e in events], float)
    out = []
    for k, e in enumerate(events):
        med = float(np.median(p[max(0, k - window):min(len(p), k + window + 1)]))
        cand = e['payload']
        if abs(cand - med) >= jump:
            while cand - med > 6:  cand -= 12
            while med - cand > 6:  cand += 12
        ne = dict(e); ne['payload'] = int(cand); out.append(ne)
    return out

def _diatonic_pcs(key):
    try:
        tonic, mode = key.tonic.pitchClass, key.mode
    except Exception:
        return set(range(12))
    iv = [0, 2, 4, 5, 7, 9, 11] if mode == 'major' else [0, 2, 3, 5, 7, 8, 10, 11]
    return {(tonic + i) % 12 for i in iv}

TRILL_MAX_UNITS = 3

def collapse_halfstep_trills(events, key):
    if len(events) < 3: return events
    pcs = _diatonic_pcs(key)
    out, i = [], 0
    while i < len(events):
        a = events[i]['payload']; b = None; j = i
        while j + 1 < len(events):
            e0, e1 = events[j], events[j + 1]
            if e1['on'] - (e0['on'] + e0['dur']) > 1: break
            if e0['dur'] > TRILL_MAX_UNITS or e1['dur'] > TRILL_MAX_UNITS: break
            if abs(e1['payload'] - e0['payload']) != 1: break
            if b is None and e1['payload'] != a: b = e1['payload']
            if e1['payload'] not in (a, b): break
            j += 1
        run = events[i:j + 1]
        ps = sorted({e['payload'] for e in run})
        if len(run) >= 3 and len(ps) == 2 and ((ps[0] % 12 in pcs) != (ps[1] % 12 in pcs)):
            keep = ps[0] if ps[0] % 12 in pcs else ps[1]
            out.append({'on': run[0]['on'],
                        'dur': (run[-1]['on'] + run[-1]['dur']) - run[0]['on'],
                        'kind': 'note', 'payload': int(keep)})
            i = j + 1
        else:
            out.append(dict(events[i])); i += 1
    return out


def collapse_chromatic_neighbors(events, key):
    if len(events) < 2: return events
    pcs = _diatonic_pcs(key)
    ev = [dict(e) for e in events]
    out, i = [], 0
    while i < len(ev):
        e = ev[i]
        if e['dur'] > CHROMA_BLIP_MAX or e['payload'] % 12 in pcs:
            out.append(e); i += 1; continue
        prv = out[-1] if out else None
        nxt = ev[i + 1] if i + 1 < len(ev) else None
        prv_ok = (prv is not None and prv['payload'] % 12 in pcs
                  and abs(e['payload'] - prv['payload']) == 1
                  and e['on'] - (prv['on'] + prv['dur']) <= 1)
        nxt_ok = (nxt is not None and nxt['payload'] % 12 in pcs
                  and abs(e['payload'] - nxt['payload']) == 1
                  and nxt['on'] - (e['on'] + e['dur']) <= 1)
        take = None
        if prv_ok and nxt_ok: take = prv if prv['dur'] >= nxt['dur'] else nxt
        elif prv_ok: take = prv
        elif nxt_ok: take = nxt
        if take is None:
            out.append(e)
        elif take is prv and e['on'] == prv['on'] + prv['dur']:
            prv['dur'] += e['dur']
        elif take is nxt and nxt['on'] == e['on'] + e['dur']:
            nxt['on'] = e['on']; nxt['dur'] += e['dur']
        else:
            e['payload'] = int(take['payload'])
            out.append(e)
        i += 1
    return out

def merge_same_pitch(events, max_gap):
    if not events: return events
    out = [dict(events[0])]
    head = out[0]['dur']
    for e in events[1:]:
        p = out[-1]
        gap = e['on'] - (p['on'] + p['dur'])
        if e['payload'] == p['payload'] and 0 <= gap <= max_gap:
            merged = (e['on'] + e['dur']) - p['on']
            head_frag = head    <= FRAGMENT_UNITS
            e_frag    = e['dur'] <= FRAGMENT_UNITS
            if head_frag and e_frag:
                ok = merged <= FRAGMENT_MERGE_MAX
            else:
                ok = head_frag != e_frag
            if ok:
                p['dur'] = merged
                continue
        out.append(dict(e))
        head = e['dur']
    return out


def merge_same_pitch_grid(events, max_gap=0):
    return merge_same_pitch(events, max_gap)

def declutter_zigzag_bars(events):
    if not events: return events
    bars = {}
    for e in events:
        bars.setdefault(e['on'] // U_PER_BAR, []).append(dict(e))
    out = []
    for bno in sorted(bars):
        notes = sorted(bars[bno], key=lambda e: e['on'])
        if len(notes) < ZIGZAG_MIN_NOTES:
            out += notes; continue
        deltas = [b['payload'] - a['payload'] for a, b in zip(notes, notes[1:])]
        flips = sum(1 for a, b in zip(deltas, deltas[1:])
                    if a != 0 and b != 0 and (a > 0) != (b > 0)
                    and abs(a) >= ZIGZAG_LEAP and abs(b) >= ZIGZAG_LEAP)
        frac = flips / max(1, len(deltas) - 1)
        if frac < ZIGZAG_FRAC:
            out += notes; continue
        cells = {}
        for e in notes:
            cells.setdefault(e['on'] // 2, []).append(e)
        kept = sorted((max(v, key=lambda e: e['dur']) for v in cells.values()),
                      key=lambda e: e['on'])
        bar_end = (bno + 1) * U_PER_BAR
        for i, e in enumerate(kept):
            on2 = (e['on'] // 2) * 2
            end2 = ((kept[i + 1]['on'] // 2) * 2 if i + 1 < len(kept)
                    else min(bar_end, max(e['on'] + e['dur'], on2 + 2)))
            e2 = dict(e); e2['on'] = on2; e2['dur'] = max(2, end2 - on2)
            out.append(e2)
    out.sort(key=lambda e: e['on'])
    return out

def lift_low_treble_runs(events):
    if not events: return events
    out = [dict(e) for e in events]
    i = 0
    while i < len(out):
        if out[i]['payload'] > TREBLE_LOW_MEDIAN:
            i += 1; continue
        j = i
        while (j + 1 < len(out)
               and out[j + 1]['payload'] <= TREBLE_LOW_MEDIAN
               and out[j + 1]['on'] - (out[j]['on'] + out[j]['dur']) < TREBLE_PHRASE_GAP):
            j += 1
        run = out[i:j + 1]
        med = float(np.median([e['payload'] for e in run]))
        eligible = (len(run) >= 3) or (med < TREBLE_TINY_LOW)
        if eligible:
            lift = 0
            while med + lift < TREBLE_LOW_MEDIAN and lift < 36:
                lift += 12
            for e in run: e['payload'] += lift
        i = j + 1
    return out

def drop_bass_collisions(cevents, vevents):
    if not vevents: return cevents
    out = []
    for c in cevents:
        nc = dict(c)
        lows = [v['payload'] for v in vevents
                if v['on'] < nc['on'] + nc['dur'] and nc['on'] < v['on'] + v['dur']]
        if lows:
            low = min(lows)
            if (low <= TREBLE_LOW_MEDIAN
                    and low <= max(nc['payload'])
                    and min(nc['payload']) - 12 >= BASS_DROP_FLOOR):
                nc['payload'] = [p - 12 for p in nc['payload']]
        out.append(nc)
    return out

def build_spelling(key):
    pc_map = {}
    for p in key.getPitches():
        pc_map[p.pitchClass] = p.name
    prefer_sharp = key.sharps >= 0
    for pc in range(12):
        if pc in pc_map: continue
        pp = m21.pitch.Pitch(midi=60 + pc)
        if prefer_sharp and pp.accidental and pp.accidental.alter < 0:
            pp = pp.getEnharmonic()
        elif (not prefer_sharp) and pp.accidental and pp.accidental.alter > 0:
            pp = pp.getEnharmonic()
        pc_map[pc] = pp.name
    return pc_map

def spelled_pitch(midi, pc_map):
    octave = midi // 12 - 1
    p = m21.pitch.Pitch(pc_map[midi % 12]); p.octave = octave
    if abs(p.midi - midi) >= 12: p.octave += (midi - p.midi) // 12
    return p

def compact_triad(midi_pitches):
    pcs = sorted(set(p % 12 for p in midi_pitches))
    root_pc = min(midi_pitches) % 12
    def nearest(target, tol):
        cands = [pc for pc in pcs if (pc - root_pc) % 12 in target]
        return cands[0] if cands else None
    third = nearest({3, 4}, 1)
    fifth = nearest({6, 7, 8}, 1)
    chosen = [root_pc] + [x for x in (third, fifth) if x is not None]
    if len(chosen) < 3:
        chosen = (chosen + [pc for pc in pcs if pc not in chosen])[:3]
    base, out, prev = BASS_OCTAVE * 12 + 12, [], None
    for pc in [root_pc] + [c for c in chosen if c != root_pc]:
        pitch = base + pc
        if prev is not None and pitch <= prev: pitch += 12
        out.append(pitch); prev = pitch
    return out

def merge_identical(chords):
    out = []
    for c in chords:
        if out and out[-1]['payload'] == c['payload'] and out[-1]['on'] + out[-1]['dur'] >= c['on']:
            out[-1]['dur'] = (c['on'] + c['dur']) - out[-1]['on']
        else:
            out.append(dict(c))
    return out

def tile_voice(events, total_u, legato=0):
    clean = []
    for i, e in enumerate(events):
        on, dur = e['on'], e['dur']
        if i < len(events) - 1 and on + dur > events[i+1]['on']:
            dur = events[i+1]['on'] - on
        if dur > 0: clean.append({**e, 'on': on, 'dur': dur})
    tiled, cursor = [], 0
    for e in clean:
        if e['on'] > cursor:
            gap = e['on'] - cursor
            if legato and gap <= legato and tiled and tiled[-1]['kind'] != 'rest':
                tiled[-1]['dur'] += gap
                cursor += gap
            else:
                tiled.append({'on': cursor, 'dur': gap, 'kind': 'rest', 'payload': None})
                cursor = e['on']
        ne = dict(e); ne['on'] = cursor
        tiled.append(ne); cursor = ne['on'] + ne['dur']
    if cursor < total_u:
        tiled.append({'on': cursor, 'dur': total_u - cursor, 'kind': 'rest', 'payload': None})
    out = []
    for e in tiled:
        on, dur = e['on'], e['dur']
        while dur > 0:
            take = min(dur, U_PER_BAR - (on % U_PER_BAR))
            if take <= 0: break
            out.append({'on': on, 'dur': take, 'kind': e['kind'], 'payload': e['payload'],
                        'tie_start': e['kind'] != 'rest' and take < dur,
                        'tie_stop':  e['kind'] != 'rest' and on != e['on']})
            on += take; dur -= take
    return out

def _tie(el, e):
    if e.get('tie_start') and e.get('tie_stop'): el.tie = m21.tie.Tie('continue')
    elif e.get('tie_start'): el.tie = m21.tie.Tie('start')
    elif e.get('tie_stop'):  el.tie = m21.tie.Tie('stop')

def apply_key_accidentals(part, key):
    keymap = {p.step: p.accidental.alter for p in key.alteredPitches}
    for m in part.getElementsByClass('Measure'):
        state = {}
        for el in m.notesAndRests:
            if el.isRest: continue
            for pp in (el.pitches if el.isChord else [el.pitch]):
                key_alter  = keymap.get(pp.step, 0.0)
                note_alter = pp.accidental.alter if pp.accidental else 0.0
                so = (pp.step, pp.octave)
                expected = state.get(so, key_alter)
                if note_alter != expected:
                    if pp.accidental is None:
                        pp.accidental = m21.pitch.Accidental('natural')
                    pp.accidental.displayStatus = True
                    state[so] = note_alter
                elif pp.accidental is not None:
                    pp.accidental.displayStatus = False
    return part

def build_part(tiled, total_u, clef, key, ts, name, pc_map, tempo=None):
    part = m21.stream.Part(); part.partName = name
    for b in range((total_u + U_PER_BAR - 1)//U_PER_BAR):
        m = m21.stream.Measure(number=b+1)
        if b == 0:
            m.insert(0, clef); m.insert(0, ts)
            if key is not None: m.insert(0, key)
            if tempo is not None: m.insert(0, m21.tempo.MetronomeMark(number=int(round(tempo))))
        bs = b*U_PER_BAR
        for e in [x for x in tiled if bs <= x['on'] < bs+U_PER_BAR]:
            ql, off = u_to_q(e['dur']), u_to_q(e['on']-bs)
            if e['kind']=='rest':
                el = m21.note.Rest(quarterLength=ql)
            elif e['kind']=='note':
                el = m21.note.Note(spelled_pitch(e['payload'], pc_map), quarterLength=ql); _tie(el,e)
            else:
                el = m21.chord.Chord([spelled_pitch(p, pc_map) for p in e['payload']], quarterLength=ql); _tie(el,e)
            m.insert(off, el)
        part.append(m)
    if key is not None:
        apply_key_accidentals(part, key)
    return part


def consolidate_melody(events, merge_gap=None):
    if merge_gap is None: merge_gap = MERGE_GAP_UNITS
    return merge_same_pitch(events, merge_gap)

def redistribute_double_dots(tiled):
    if not tiled: return tiled
    out = [dict(e) for e in tiled]
    for i in range(len(out) - 1):
        a, e = out[i], out[i + 1]
        if a['dur'] not in DD_FIX: continue
        if a['on'] + a['dur'] != e['on']: continue
        if a['on'] // U_PER_BAR != e['on'] // U_PER_BAR: continue
        if (a.get('tie_start') or a.get('tie_stop')
                or e.get('tie_start') or e.get('tie_stop')): continue
        shift = a['dur'] - DD_FIX[a['dur']]
        give_clean = (e['dur'] + shift) in CLEAN_DURS
        take_clean = (e['dur'] - shift) >= 1 and (e['dur'] - shift) in CLEAN_DURS
        if give_clean or not take_clean:
            a['dur'] = DD_FIX[a['dur']]
            e['on'] -= shift; e['dur'] += shift
        else:
            a['dur'] += shift
            e['on'] += shift; e['dur'] -= shift
    return out

def absorb_micro_rests(tiled):
    ev = [dict(e) for e in tiled]
    def same_bar(a, b): return a['on'] // U_PER_BAR == b['on'] // U_PER_BAR
    def tied(e): return e.get('tie_start') or e.get('tie_stop')

    changed = True
    while changed:
        changed = False
        for i, e in enumerate(ev):
            if e['kind'] != 'rest': continue
            prv = ev[i - 1] if i > 0 else None
            nxt = ev[i + 1] if i + 1 < len(ev) else None
            prv_ok = (prv is not None and prv['kind'] == 'note' and not tied(prv)
                      and same_bar(prv, e) and prv['on'] + prv['dur'] == e['on'])
            nxt_ok = (nxt is not None and nxt['kind'] == 'note' and not tied(nxt)
                      and same_bar(e, nxt) and e['on'] + e['dur'] == nxt['on'])
            if e['dur'] == 1:
                if prv_ok and prv['dur'] + 1 in CLEAN_DURS:
                    prv['dur'] += 1; del ev[i]; changed = True; break
                if nxt_ok and nxt['dur'] + 1 in CLEAN_DURS:
                    nxt['on'] -= 1; nxt['dur'] += 1; del ev[i]; changed = True; break
                if prv_ok and prv['dur'] + 1 not in DD_FIX:
                    prv['dur'] += 1; del ev[i]; changed = True; break
                if nxt_ok and nxt['dur'] + 1 not in DD_FIX:
                    nxt['on'] -= 1; nxt['dur'] += 1; del ev[i]; changed = True; break
            elif e['dur'] % 2 == 1:
                if prv_ok and prv['dur'] % 2 == 1 and prv['dur'] + 1 not in DD_FIX:
                    prv['dur'] += 1; e['on'] += 1; e['dur'] -= 1; changed = True; break
                if nxt_ok and nxt['dur'] % 2 == 1 and nxt['dur'] + 1 not in DD_FIX:
                    e['dur'] -= 1; nxt['on'] -= 1; nxt['dur'] += 1; changed = True; break
    return ev

def even_out_snapped_pairs(tiled):
    ev = [dict(e) for e in tiled]
    for i in range(len(ev) - 1):
        a, b = ev[i], ev[i + 1]
        if a['kind'] != 'note' or b['kind'] != 'note': continue
        if a['on'] + a['dur'] != b['on']: continue
        if a['on'] // U_PER_BAR != b['on'] // U_PER_BAR: continue
        if (a.get('tie_start') or a.get('tie_stop')
                or b.get('tie_start') or b.get('tie_stop')): continue
        if {a['dur'], b['dur']} == {1, 3}:
            if a['dur'] == 1: b['on'] += 1
            else:             b['on'] -= 1
            a['dur'] = b['dur'] = 2
    return ev

def measure_health(score):
    probs = []
    for p in score.parts:
        pname = (p.partName or "?")
        for m in p.getElementsByClass('Measure'):
            evs = sorted(((float(e.offset), float(e.quarterLength))
                          for e in m.notesAndRests), key=lambda x: x[0])
            cur, bad = 0.0, None
            for off, ql in evs:
                if off > cur + 1e-6: bad = f"gap at beat {cur:g}"; break
                if off < cur - 1e-6: bad = f"overlap at beat {off:g}"; break
                cur = off + ql
            if bad is None and abs(cur - 4.0) > 1e-6: bad = f"sums to {cur:g} beats"
            if bad: probs.append((pname, m.number, bad))
    return probs

def _beat_mapper(beat_times):
    if beat_times is None: return None
    bt = np.asarray(beat_times, float)
    if len(bt) < 2: return None
    idx = np.arange(len(bt), dtype=float)
    d0 = max(bt[1] - bt[0], 1e-6); d1 = max(bt[-1] - bt[-2], 1e-6)
    def to_beats(t):
        t = float(t)
        if t <= bt[0]:  return (t - bt[0]) / d0
        if t >= bt[-1]: return (len(bt) - 1.0) + (t - bt[-1]) / d1
        return float(np.interp(t, bt, idx))
    return to_beats

def grand_staff_make(vocal_midi, chord_midi, out_xml, rng=None, beat_times=None,
                     tempo=None):
    cs = m21.converter.parse(chord_midi)
    key = cs.analyze('key')

    vpm = pretty_midi.PrettyMIDI(vocal_midi)
    _vt, _vtempi = vpm.get_tempo_changes()
    bpm = float(tempo) if tempo else (float(_vtempi[0]) if len(_vtempi) else 128.0)
    bmap = _beat_mapper(beat_times)
    def _u(t):
        if bmap is not None: return bmap(t) * U_PER_BEAT
        return t * bpm / 60.0 * U_PER_BEAT
    raw = sorted((n for inst in vpm.instruments for n in inst.notes),
                 key=lambda n: (n.start, -n.pitch))
    vevents = []
    for i, n in enumerate(raw):
        end = n.end
        if i + 1 < len(raw) and raw[i + 1].start < end:
            end = raw[i + 1].start
        if end <= n.start: continue
        a = _u(n.start); b = _u(end)
        if rng is not None:
            a += rng.uniform(-SEED_MELODY_JITTER, SEED_MELODY_JITTER)
            b += rng.uniform(-SEED_MELODY_JITTER, SEED_MELODY_JITTER)
        on = int(round(a)); dur = max(1, int(round(b)) - on)
        if on < 0: on = 0
        vevents.append({'on': on, 'dur': dur, 'kind': 'note', 'payload': int(n.pitch)})
    vevents.sort(key=lambda e: e['on'])

    kt = [("mid", len(vevents))]
    def _kt(name, ev): kt.append((name, len(ev))); return ev
    vevents = _kt("fold",   fold_octave_outliers_events(vevents))
    vevents = _kt("trills", collapse_halfstep_trills(vevents, key))
    vevents = _kt("blips",  collapse_chromatic_neighbors(vevents, key))
    vevents = _kt("consol", consolidate_melody(vevents))
    vevents = _kt("rhythm", clean_melody_rhythm(vevents))
    vevents = _kt("samepitch", merge_same_pitch_grid(vevents))
    vevents = _kt("zigzag", declutter_zigzag_bars(vevents))
    vevents = _kt("lift",   lift_low_treble_runs(vevents))

    cpm = pretty_midi.PrettyMIDI(chord_midi)
    cgroups = {}
    for i2 in cpm.instruments:
        for n2 in i2.notes:
            cgroups.setdefault(round(n2.start, 3), []).append(n2)
    craw = []
    for st in sorted(cgroups):
        grp = cgroups[st]
        en = max(n2.end for n2 in grp)
        on  = snap(max(0, int(round(_u(st)))), CHORD_SNAP_UNITS)
        end = snap(int(round(_u(en))), CHORD_SNAP_UNITS)
        craw.append({'on': on, 'dur': max(CHORD_SNAP_UNITS, end - on),
                     'kind': 'chord', 'payload': compact_triad([n2.pitch for n2 in grp])})
    cevents = merge_identical(craw)
    cevents = drop_bass_collisions(cevents, vevents)

    total_u = max([e['on']+e['dur'] for e in vevents+cevents] + [U_PER_BAR])
    total_u = ((total_u + U_PER_BAR - 1)//U_PER_BAR)*U_PER_BAR
    if total_u > 50000:
        raise ValueError(f"Refusing to build {total_u//U_PER_BAR} measures: a MIDI note "
                         f"carries an implausible timestamp. Check the input MIDI files "
                         f"for stray notes.")

    pc_map = build_spelling(key)
    vt = tile_voice(vevents, total_u, legato=LEGATO_FILL_UNITS)
    vt = absorb_micro_rests(vt)
    vt = even_out_snapped_pairs(vt)
    vt = redistribute_double_dots(vt)
    vt = absorb_micro_rests(vt)
    ct = tile_voice(cevents, total_u, legato=0)
    score = m21.stream.Score()
    score.insert(0, build_part(vt, total_u, m21.clef.TrebleClef(), key, m21.meter.TimeSignature('4/4'),"Right Hand", pc_map, tempo=bpm))
    score.insert(0, build_part(ct, total_u, m21.clef.BassClef(),  key, m21.meter.TimeSignature('4/4'),"Left Hand", pc_map))
    os.makedirs(os.path.dirname(out_xml) or ".", exist_ok=True)
    score.write('musicxml', fp=out_xml)
    kt.append(("page", len([e for e in vt if e['kind'] == 'note'])))
    deltas = [f"{kt[0][0]} {kt[0][1]}"] + [
        f"{n} {c}" + (f"({c - p:+d})" if c != p else "")
        for (n, c), (_, p) in zip(kt[1:], kt[:-1])]
    print("   melody notes by pass: " + " | ".join(deltas))
    probs = measure_health(score)
    for pn, mn, why in probs[:6]:
        print(f"   !! {pn} measure {mn}: {why}")
    return out_xml, key.name, bpm, len(probs), score


# stage 0 demucs
def run_demucs(input_audio):
    song = os.path.splitext(os.path.basename(input_audio))[0]
    stem_dir = os.path.join(STEM_ROOT, DEMUCS_MODEL, song)
    needed = ["vocals.wav", "bass.wav", "drums.wav", "other.wav"]
    if all(os.path.exists(os.path.join(stem_dir, s)) for s in needed):
        print(f"   stems already present in {stem_dir}/ -> skipping Demucs")
        return stem_dir, song
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"input audio not found: {input_audio}")
    cmd = ["demucs", "-n", DEMUCS_MODEL, "-o", STEM_ROOT, input_audio]
    print(f"   {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError("Demucs failed (see output above)")
    missing = [s for s in needed if not os.path.exists(os.path.join(stem_dir, s))]
    if missing:
        raise RuntimeError(f"Demucs finished but stems missing: {missing} in {stem_dir}")
    return stem_dir, song


# stage 1 accompaniment mix
def run_accompaniment(stem_dir, out_wav):
    bass, _  = librosa.load(os.path.join(stem_dir, "bass.wav"),  sr=SR, mono=True)
    drums, _ = librosa.load(os.path.join(stem_dir, "drums.wav"), sr=SR, mono=True)
    other, _ = librosa.load(os.path.join(stem_dir, "other.wav"), sr=SR, mono=True)
    m = min(len(bass), len(drums), len(other))
    acc = bass[:m] + drums[:m] + other[:m]
    acc = acc / (np.max(np.abs(acc)) + 1e-12) * 0.9
    sf.write(out_wav, acc, SR)
    print(f"   {out_wav}  ({m/SR:.1f}s)")


# stage 2 beats
def fit_uniform_grid(beat_times, tol=GRID_FIT_TOL, need=GRID_FIT_COVERAGE):
    bt = np.asarray(beat_times, float)
    if len(bt) < 16:
        return bt, False, 0.0, 0.0
    iv = np.diff(bt)
    seed = float(np.percentile(iv, 20))
    if seed <= 1e-6:
        return bt, False, 0.0, 0.0
    cand = np.linspace(0.85 * seed, 1.25 * seed, 4000)
    votes = np.exp(2j * np.pi * bt[None, :] / cand[:, None]).mean(axis=1)
    best = int(np.argmax(np.abs(votes)))
    tau = float(cand[best])
    phase = float(np.angle(votes[best]) / (2 * np.pi) * tau)
    for _ in range(3):
        k = np.round((bt - phase) / tau)
        A = np.vstack([k, np.ones_like(k)]).T
        tau, phase = np.linalg.lstsq(A, bt, rcond=None)[0]
        tau, phase = float(tau), float(phase)
        if tau <= 1e-6:
            return bt, False, 0.0, 0.0
    k = np.round((bt - phase) / tau)
    coverage = float(np.mean(np.abs(bt - (phase + k * tau)) <= tol * tau))
    if coverage < need:
        return bt, False, coverage, 60.0 / tau
    k0 = int(round((bt[0] - phase) / tau))
    k1 = int(round((bt[-1] - phase) / tau))
    return phase + np.arange(k0, k1 + 1) * tau, True, coverage, 60.0 / tau


def repair_beat_grid(beat_times, tol=BEAT_GAP_TOL, win=BEAT_LOCAL_WIN):
    bt = np.asarray(beat_times, float)
    if len(bt) < 4:
        return bt, 0
    iv = np.diff(bt)
    glob = float(np.median(iv))
    out, added, h = [bt[0]], 0, win // 2
    for i, gap in enumerate(iv):
        lo, hi = max(0, i - h), min(len(iv), i + h + 1)
        local = float(np.median(iv[lo:hi])) or glob
        k = int(round(gap / local)) if local > 1e-9 else 1
        if k >= 2 and abs(gap / local - k) <= tol:
            step = gap / k
            out.extend(bt[i] + j * step for j in range(1, k))
            added += k - 1
        out.append(bt[i + 1])
    return np.asarray(out, float), added


def regularize_beat_grid(beat_times, max_dev=BEAT_STEADY_MAX, rms_dev=BEAT_STEADY_RMS):
    bt = np.asarray(beat_times, float)
    if len(bt) < 8:
        return bt, False, 0.0, 0.0
    idx = np.arange(len(bt), dtype=float)
    slope, intercept = np.polyfit(bt, idx, 1)
    if slope <= 1e-9:
        return bt, False, 0.0, 0.0
    resid = idx - (slope * bt + intercept)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    mx  = float(np.abs(resid).max())
    if mx <= max_dev and rms <= rms_dev:
        return (idx - intercept) / slope, True, rms, mx
    return bt, False, rms, mx


def detect_tempo_and_beats(accomp_wav, cache_npz=None):
    if cache_npz and os.path.exists(cache_npz):
        c = np.load(cache_npz, allow_pickle=True)
        if "repair" in c.files and int(c["repair"]) == BEAT_REPAIR_VERSION:
            print(f"   reuse cached tempo ({float(c['tempo']):.1f} BPM, "
                  f"{len(c['beats'])} beats)")
            return float(c['tempo']), np.asarray(c['beats'], dtype=float), False
        tempo = float(c['tempo']); beat_times = np.asarray(c['beats'], dtype=float)
        print(f"   cached grid predates the repair ({tempo:.1f} BPM, "
              f"{len(beat_times)} beats) -> checking it")
        return _finish_beats(beat_times, cache_npz, was_cached=True)
    try:
        if HAS_MADMOM:
            proc = madmom.features.beats.RNNBeatProcessor()(accomp_wav)
            beat_times = madmom.features.beats.BeatTrackingProcessor(fps=100)(proc)
            tempo = 60.0 / np.median(np.diff(beat_times))
            print(f"   madmom RNN: ~{tempo:.1f} BPM, {len(beat_times)} beats")
        else:
            y, _ = librosa.load(accomp_wav, sr=SR, mono=True)
            tarr, bframes = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, trim=False)
            tempo = float(np.atleast_1d(tarr)[0])
            beat_times = librosa.frames_to_time(bframes, sr=SR, hop_length=HOP)
            print(f"   librosa: {tempo:.1f} BPM, {len(beat_times)} beats")
    except Exception as e:
        y, _ = librosa.load(accomp_wav, sr=SR, mono=True)
        dur = len(y) / SR
        tempo = 120.0
        beat_times = np.arange(0, dur, 0.5)
        print(f"   beat tracking failed ({e}) -> 120 BPM fallback")
    return _finish_beats(np.asarray(beat_times, dtype=float), cache_npz)


def _finish_beats(beat_times, cache_npz=None, was_cached=False):
    before = len(beat_times)
    grid, fitted, coverage, fit_bpm = fit_uniform_grid(beat_times)
    if fitted:
        beat_times = grid
        print(f"   one fixed pulse explains {coverage*100:.0f}% of the tracked beats "
              f"-> rebuilt on it")
        print(f"   grid: {before} -> {len(beat_times)} beats "
              f"({len(beat_times) - before:+d}), {fit_bpm:.2f} BPM")
        changed = True
    else:
        print(f"   no single pulse fits (best explains {coverage*100:.0f}%) "
              f"-> tempo moves, following the tracked beats")
        beat_times, added = repair_beat_grid(beat_times)
        if added:
            print(f"   gap repair: {added} beat(s) the tracker missed, "
                  f"{before} -> {len(beat_times)}")
        beat_times, straight, _rms, mx = regularize_beat_grid(beat_times)
        if straight:
            print(f"   beats sit within {mx:.2f} of a line -> using the line")
        changed = bool(added or straight)
        print(f"   grid: {len(beat_times)} beats, "
              f"{60.0 / np.median(np.diff(beat_times)):.2f} BPM")
    iv = np.diff(beat_times)
    tempo = float(60.0 / np.median(iv)) if len(iv) else 120.0
    if cache_npz:
        os.makedirs(os.path.dirname(cache_npz) or ".", exist_ok=True)
        np.savez(cache_npz, tempo=tempo, beats=beat_times,
                 repair=BEAT_REPAIR_VERSION)
    return tempo, beat_times, (was_cached and changed)


# stage 3 crepe
def run_crepe(vocals_wav, cache_npz):
    import torch, torchcrepe
    src_mtime = os.path.getmtime(vocals_wav)
    if os.path.exists(cache_npz):
        c = np.load(cache_npz, allow_pickle=True)
        if (str(c.get("model")) == CREPE_MODEL and float(c.get("mtime")) == src_mtime
                and int(c.get("hop")) == HOP):
            print(f"   loaded cached CREPE ({len(c['f0'])} frames)")
            return c["times"], c["f0"], c["conf"]
    vocals, _ = librosa.load(vocals_wav, sr=SR, mono=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunk = SR * CREPE_CHUNK_SEC
    n = int(np.ceil(len(vocals) / chunk))
    f0a, cfa = [], []
    for i in range(n):
        s, e = i * chunk, min((i + 1) * chunk, len(vocals))
        p, per = torchcrepe.predict(
            torch.tensor(vocals[s:e]).float().unsqueeze(0),
            SR, hop_length=HOP, fmin=CREPE_FMIN, fmax=CREPE_FMAX,
            model=CREPE_MODEL, device=device, return_periodicity=True)
        f0a.append(np.atleast_1d(p.squeeze().cpu().numpy()))
        cfa.append(np.atleast_1d(per.squeeze().cpu().numpy()))
        print(f"   CREPE chunk {i+1}/{n}")
    f0 = np.concatenate(f0a); conf = np.concatenate(cfa)
    times = np.arange(len(f0)) * HOP / SR
    os.makedirs(os.path.dirname(cache_npz) or ".", exist_ok=True)
    np.savez(cache_npz, times=times, f0=f0, conf=conf,
             model=CREPE_MODEL, mtime=src_mtime, hop=HOP)
    return times, f0, conf


# stage 4 vocal notes
def aligned_rms(vocals, n_frames):
    rms = librosa.feature.rms(y=vocals, frame_length=HOP * 4, hop_length=HOP)[0]
    if len(rms) < n_frames:
        rms = np.pad(rms, (0, n_frames - len(rms)), mode='edge')
    return rms[:n_frames]

def run_vocals(vocals_wav, cache_npz, out_midi, tempo):
    times, f0, conf = run_crepe(vocals_wav, cache_npz)
    vocals, _ = librosa.load(vocals_wav, sr=SR, mono=True)
    rms = aligned_rms(vocals, len(f0))
    notes = extract_notes_fused_v2(
        times, f0, conf, rms,
        conf_thresh=V_CONF_THRESH, pitch_tol=V_PITCH_TOL,
        min_frames=V_MIN_FRAMES, bridge_frames=V_BRIDGE_FRAMES,
        depth_ratio=V_DEPTH_RATIO)
    notes = fix_octave_outliers(notes)
    notes = fix_intro_octaves(notes)
    kept = [(on, off, p) for (on, off, p) in notes if V_MIDI_LO <= p <= V_MIDI_HI and off > on]
    clean_tempo = int(round(tempo))
    pm = pretty_midi.PrettyMIDI(initial_tempo=clean_tempo)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    inst = pretty_midi.Instrument(program=52, name="Vocals")
    for on, off, p in kept:
        inst.notes.append(pretty_midi.Note(velocity=85, pitch=int(max(0, min(127, p))),
                                           start=float(on), end=float(off)))
    pm.instruments.append(inst)
    os.makedirs(os.path.dirname(out_midi) or ".", exist_ok=True)
    pm.write(out_midi)
    print(f"   {len(kept)} notes -> {out_midi}  (stamped {clean_tempo} BPM)")
    return len(kept)


# stage 6 note tails
def run_extend(in_midi, vocals_wav, out_midi):
    y, _  = librosa.load(vocals_wav, sr=SR, mono=True)
    rms   = librosa.feature.rms(y=y, frame_length=HOP * 4, hop_length=HOP)[0]
    t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=HOP)
    def energy_between(a, b):
        m = (t_rms >= a) & (t_rms < b)
        return float(np.median(rms[m])) if m.any() else 0.0
    pm    = pretty_midi.PrettyMIDI(in_midi)
    inst  = pm.instruments[0]
    notes = sorted(inst.notes, key=lambda n: n.start)
    note_e = [e for e in (energy_between(n.start, n.end) for n in notes) if e > 0]
    hold_thresh = (np.median(note_e) * EXT_HOLD_FRAC) if note_e else 0.0
    extended = 0
    for i in range(len(notes) - 1):
        cur, nxt = notes[i], notes[i + 1]
        gap = nxt.start - cur.end
        if gap <= 0:
            continue
        e = energy_between(cur.end, nxt.start)
        if e >= hold_thresh and gap <= EXT_MAX_FILL_SEC:
            cur.end = nxt.start; extended += 1
    os.makedirs(os.path.dirname(out_midi) or ".", exist_ok=True)
    pm.write(out_midi)
    print(f"   extended {extended} clipped notes -> {out_midi}")


# stage 8 chords
def run_chords(bass_wav, other_wav, accomp_wav, out_midi, out_txt, tempo, beat_times):
    bass, _  = librosa.load(bass_wav,  sr=SR, mono=True)
    other, _ = librosa.load(other_wav, sr=SR, mono=True)
    lens = [len(bass), len(other)]
    if os.path.exists(accomp_wav):
        accomp, _ = librosa.load(accomp_wav, sr=SR, mono=True)
        lens.append(len(accomp))
    min_len = min(lens)
    bass, other = bass[:min_len], other[:min_len]
    duration = min_len / SR

    clean_tempo = int(round(tempo))
    beat_dur = 60.0 / clean_tempo
    bt = np.asarray(beat_times, dtype=float)
    if len(bt) == 0 or bt[0] > 0.05:
        bt = np.insert(bt, 0, 0.0)
    while bt[-1] < duration:
        bt = np.append(bt, bt[-1] + beat_dur)

    beat_intervals = list(zip(bt[:-1], bt[1:]))
    chroma_other = librosa.feature.chroma_cens(y=other, sr=SR, hop_length=CH_HOP)
    chroma_bass  = librosa.feature.chroma_cqt(y=bass, sr=SR, hop_length=CH_HOP)
    templates = generate_templates()
    names = list(templates.keys())

    raw_chords = []
    for b_start, b_end in beat_intervals:
        c = match_chord(chroma_other, chroma_bass, b_start, b_end, templates, names, SR, CH_HOP)
        if c is not None:
            raw_chords.append({'start': b_start, 'end': b_end, 'name': c['name'], 'margin': c['margin']})
        elif raw_chords:
            raw_chords.append({**raw_chords[-1], 'start': b_start, 'end': b_end})

    merged = []
    for c in raw_chords:
        if merged and merged[-1]['name'] == c['name']:
            merged[-1]['end'] = c['end']; merged[-1]['margins'].append(c['margin'])
        else:
            merged.append({'start': c['start'], 'end': c['end'], 'name': c['name'], 'margins': [c['margin']]})

    merged, chd_fixed = fix_nondiatonic_blocks(merged, templates, beat_dur)

    grid_unit = beat_dur / 4
    for c in merged:
        c['start'] = snap_to_grid(c['start'], grid_unit)
        c['end']   = snap_to_grid(c['end'],   grid_unit)
    for i in range(1, len(merged)):
        if merged[i]['start'] < merged[i - 1]['end']:
            merged[i]['start'] = merged[i - 1]['end']
    for c in merged:
        if c['end'] <= c['start']:
            c['end'] = c['start'] + grid_unit

    pm = pretty_midi.PrettyMIDI(initial_tempo=clean_tempo)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    inst = pretty_midi.Instrument(program=0, name="Chords")
    os.makedirs(os.path.dirname(out_txt)  or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_midi) or ".", exist_ok=True)
    with open(out_txt, "w") as f:
        for c in merged:
            visual_end = max(c['start'] + grid_unit, c['end'] - 0.01)
            for p in templates[c['name']]['midi']:
                inst.notes.append(pretty_midi.Note(velocity=65, pitch=p,
                                                   start=float(c['start']), end=float(visual_end)))
            f.write(f"{c['start']:.3f},{c['end']:.3f},{c['name']}\n")
    pm.instruments.append(inst)
    pm.write(out_midi)
    print(f"   {len(merged)} chord blocks | diatonic guard fixed {chd_fixed} stray -> {out_midi}  (stamped {clean_tempo} BPM)")


# basic pitch melody line
from scipy.signal import stft, resample_poly

CONF_FLOOR   = 0.25
MIN_DUR      = 0.08
MELODY_FLOOR = 48
W_PITCH      = 0.40
W_GAP        = 0.30
GAP_CAP      = 1.0
MAX_LINK_SEC = 8.0
NOTE_COST    = 0.10
REG_BONUS    = 0.20
REG_WIN      = 3.0
OVERLAP_TOL  = 0.15
W_SAL        = 0.60
GHOST_W      = 0.50
SAL_LOCAL    = 0.50
SAL_SR       = 16000
SAL_NFFT     = 4096
SAL_HOP      = 512

def run_basic_pitch(other_wav, cache_npz):
    if os.path.exists(cache_npz):
        c = np.load(cache_npz)
        if float(c['mtime']) == os.path.getmtime(other_wav):
            print(f"   loaded cached basic-pitch ({len(c['s'])} notes)")
            return list(zip(c['s'], c['e'], c['p'].astype(int), c['a']))
    from basic_pitch.inference import predict
    print(f"   running basic-pitch on {os.path.basename(other_wav)} (takes a bit)...")
    _, _, ev = predict(other_wav)
    notes = [(float(s), float(e), int(p), float(a)) for (s, e, p, a, *_) in ev]
    os.makedirs(os.path.dirname(cache_npz) or ".", exist_ok=True)
    np.savez(cache_npz,
             s=np.array([n[0] for n in notes]), e=np.array([n[1] for n in notes]),
             p=np.array([n[2] for n in notes]), a=np.array([n[3] for n in notes]),
             mtime=os.path.getmtime(other_wav))
    print(f"   basic-pitch found {len(notes)} raw notes (cached)")
    return notes

def _salience_core(y, sr, notes):
    if sr != SAL_SR:
        g = int(np.gcd(int(sr), SAL_SR))
        y = resample_poly(y, SAL_SR // g, int(sr) // g)
        sr = SAL_SR
    f, t, Z = stft(y, fs=sr, nperseg=SAL_NFFT, noverlap=SAL_NFFT - SAL_HOP, padded=True)
    S = np.abs(Z)
    nb = len(f)
    def band(f0):
        lo = int(np.searchsorted(f, f0 * 2 ** (-0.5 / 12)))
        hi = int(np.searchsorted(f, f0 * 2 ** (0.5 / 12)))
        lo = min(lo, nb - 1); hi = max(hi, lo + 1); hi = min(hi, nb)
        return lo, hi
    raw = np.zeros(len(notes))
    for k, (s, e, p, a) in enumerate(notes):
        i0 = int(np.searchsorted(t, s)); i1 = max(i0 + 1, int(np.searchsorted(t, e)))
        if i0 >= S.shape[1]:
            continue
        i1 = min(i1, S.shape[1])
        f0 = 440.0 * 2 ** ((p - 69) / 12.0)
        l1, h1 = band(f0)
        v = S[l1:h1, i0:i1].sum(axis=0).mean()
        if 2 * f0 < f[-1]:
            l2, h2 = band(2 * f0)
            v += 0.5 * S[l2:h2, i0:i1].sum(axis=0).mean()
        if f0 / 2 > f[1]:
            l0, h0 = band(f0 / 2)
            v -= GHOST_W * S[l0:h0, i0:i1].sum(axis=0).mean()
        raw[k] = max(v, 0.0)
    return raw

def _local_dominance(notes, raw):
    n = len(notes)
    starts = np.array([x[0] for x in notes]); ends = np.array([x[1] for x in notes])
    order = np.argsort(starts)
    so, eo, ro = starts[order], ends[order], np.asarray(raw, float)[order]
    maxdur = float((eo - so).max()) if n else 0.0
    dom = np.zeros(n)
    for oi in range(n):
        s, e = so[oi], eo[oi]
        best = ro[oi]
        k = oi - 1
        while k >= 0 and so[k] > s - maxdur:
            if eo[k] > s: best = max(best, ro[k])
            k -= 1
        k = oi + 1
        while k < n and so[k] < e:
            if eo[k] > s: best = max(best, ro[k])
            k += 1
        dom[order[oi]] = ro[order[oi]] / (best + 1e-12)
    return dom

def salience_from_audio(y, sr, notes):
    raw = _salience_core(y, sr, notes)
    global_rank = np.argsort(np.argsort(raw)) / max(len(notes) - 1, 1)
    dom = _local_dominance(notes, raw)
    return (1.0 - SAL_LOCAL) * global_rank + SAL_LOCAL * dom

def note_salience(other_wav, notes):
    y, sr = sf.read(other_wav)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return salience_from_audio(np.asarray(y, float), sr, notes)

def fold_octave_outliers(line, window=5, jump=11):
    if len(line) < 3: return line
    p = np.array([x[2] for x in line], float)
    center = np.array([np.median(p[max(0, k - window):min(len(p), k + window + 1)])
                       for k in range(len(p))])
    out = []
    for k, (s, e, pk) in enumerate(line):
        med = center[k]; cand = pk
        if abs(pk - med) >= jump:
            while cand - med > 6: cand -= 12
            while med - cand > 6: cand += 12
        out.append((s, e, int(cand)))
    return out

def dp_melody(notes, salience=None):
    pool = [(k, n) for k, n in enumerate(notes)
            if n[3] >= CONF_FLOOR and (n[1] - n[0]) >= MIN_DUR and n[2] >= MELODY_FLOOR]
    if not pool:
        return []
    pool.sort(key=lambda kn: (kn[1][0], -kn[1][3]))
    cand = [n for _, n in pool]
    sal = (np.array([salience[k] for k, _ in pool], float)
           if salience is not None else None)

    starts = np.array([n[0] for n in cand])
    pit    = np.array([n[2] for n in cand], float)
    lo_idx = np.searchsorted(starts, starts - REG_WIN, side='left')
    hi_idx = np.searchsorted(starts, starts + REG_WIN, side='right')
    center = np.array([np.median(pit[lo_idx[k]:hi_idx[k]]) for k in range(len(cand))])

    def reward(k):
        n = cand[k]
        amp = n[3] if sal is None else (1.0 - W_SAL) * n[3] + W_SAL * sal[k]
        reg = max(-1.0, min(1.0, (n[2] - center[k]) / 12.0))
        return amp + REG_BONUS * reg - NOTE_COST

    n = len(cand)
    score = [reward(k) for k in range(n)]
    prev  = [-1] * n
    lookback = MAX_LINK_SEC + max(e - s for (s, e, _, _) in cand)
    for i in range(n):
        si = cand[i][0]; pi = cand[i][2]; ri = reward(i)
        j = i - 1
        while j >= 0 and cand[j][0] > si - lookback:
            sj, ej, pj, _ = cand[j]
            if sj < si and (si - ej) <= MAX_LINK_SEC and si >= ej - OVERLAP_TOL:
                gap = max(0.0, si - ej)
                cost = W_PITCH * abs(pi - pj) / 12.0 + W_GAP * min(gap, GAP_CAP)
                cd = score[j] + ri - cost
                if cd > score[i]:
                    score[i] = cd; prev[i] = j
            j -= 1
    k = int(np.argmax(score)); path = []
    while k != -1:
        path.append(cand[k]); k = prev[k]
    path.reverse()
    out = []
    for idx, (s, e, p, a) in enumerate(path):
        if idx + 1 < len(path):
            e = min(e, path[idx + 1][0])
        if e - s >= MIN_DUR:
            out.append((s, e, p))
    return fold_octave_outliers(out)


# stage 5 basic pitch recovery
def _merge_spans(regs):
    out = []
    for r0, r1 in sorted(regs):
        if out and r0 <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], r1))
        else:
            out.append((r0, r1))
    return out

def _thin_region(vnotes, r0, r1, beat_sec):
    inside = [n for n in vnotes if r0 <= (n[0] + n[1]) / 2 < r1]
    if not inside: return vnotes, 0, 0
    keep = []
    k = r0
    while k < r1:
        w = [n for n in inside if k <= (n[0] + n[1]) / 2 < k + beat_sec]
        if w: keep.append(max(w, key=lambda n: n[1] - n[0]))
        k += beat_sec
    out = [n for n in vnotes if not (r0 <= (n[0] + n[1]) / 2 < r1)] + keep
    out.sort()
    return out, len(inside), len(keep)

def _low_conf_regions(t, conf, bpm, thresh=None, min_beats=None):
    if thresh is None: thresh = VPATCH_CONF
    if min_beats is None: min_beats = VPATCH_MIN_BEATS
    t = np.asarray(t, float); conf = np.asarray(conf, float)
    if len(t) < 2: return []
    beat = 60.0 / bpm
    nb = int(float(t[-1]) / beat) + 1
    means = []
    for k in range(nb):
        m = (t >= k * beat) & (t < (k + 1) * beat)
        voiced = m & (conf >= thresh)
        if voiced.any():
            means.append(float(conf[voiced].mean()))
        elif m.any():
            means.append(float(conf[m].mean()))
        else:
            means.append(1.0)
    regs, k = [], 0
    while k < nb:
        if means[k] < thresh:
            j = k
            while j + 1 < nb and means[j + 1] < thresh: j += 1
            if (j - k + 1) >= min_beats:
                regs.append((k * beat, (j + 1) * beat))
            k = j + 1
        else:
            k += 1
    return regs

def _replace_regions(vnotes, line, regions, min_cover=0.4):
    out = list(vnotes); swapped = []
    for (r0, r1) in regions:
        cand = []
        for (s, e, p) in line:
            s2, e2 = max(s, r0), min(e, r1)
            if e2 - s2 >= MIN_DUR: cand.append((s2, e2, int(p)))
        cover = sum(e - s for s, e, _ in cand) / max(r1 - r0, 1e-9)
        if cover < min_cover: continue
        keep = [n for n in out if not (r0 <= (n[0] + n[1]) / 2 < r1)]
        swapped.append((r0, r1, len(out) - len(keep), len(cand)))
        out = keep + cand
    out.sort()
    return out, swapped

def _patch_holes(vnotes, line, gate_sec, song_end, is_vocal=None):
    holes, skipped = [], []
    if vnotes:
        if vnotes[0][0] >= gate_sec: holes.append((0.0, vnotes[0][0]))
        for a, b in zip(vnotes, vnotes[1:]):
            if b[0] - a[1] >= gate_sec: holes.append((a[1], b[0]))
    elif song_end > 0:
        holes.append((0.0, song_end))
    if is_vocal is not None:
        keep = []
        for h in holes:
            (keep if is_vocal(*h) else skipped).append(h)
        holes = keep
    added = []
    for h0, h1 in holes:
        for (s, e, p) in line:
            s2, e2 = max(s, h0), min(e, h1)
            if e2 - s2 >= MIN_DUR:
                added.append((s2, e2, int(p)))
    return holes, added, skipped


def vocal_presence(vocals_wav, other_wav=None):
    try:
        yv, _ = librosa.load(vocals_wav, sr=SR, mono=True)
        rv = librosa.feature.rms(y=yv, frame_length=HOP * 4, hop_length=HOP)[0]
        tv = librosa.frames_to_time(np.arange(len(rv)), sr=SR, hop_length=HOP)
    except Exception as ex:
        print(f"   vocal presence test unavailable ({ex}) -> every span treated as sung")
        return None
    loud = rv[rv > np.median(rv)]
    floor = float(np.median(loud)) * VOCAL_FLOOR_FRAC if len(loud) else 0.0
    ro = to = None
    if other_wav and os.path.isfile(other_wav):
        try:
            yo, _ = librosa.load(other_wav, sr=SR, mono=True)
            ro = librosa.feature.rms(y=yo, frame_length=HOP * 4, hop_length=HOP)[0]
            to = librosa.frames_to_time(np.arange(len(ro)), sr=SR, hop_length=HOP)
        except Exception:
            ro = to = None

    def is_vocal(a, b):
        m = (tv >= a) & (tv < b)
        if not m.any(): return False
        if float(np.mean(rv[m] >= floor)) < VOCAL_ACTIVE_MIN:
            return False
        if ro is not None:
            mo = (to >= a) & (to < b)
            if mo.any():
                v = float(np.median(rv[m])); o = float(np.median(ro[mo]))
                if o > 0 and v / o < VOCAL_OTHER_SHARE:
                    return False
        return True
    return is_vocal

def run_vocal_patch(in_mid, vocals_wav, bp_cache_v, out_mid, crepe_npz=None,
                    other_wav=None):
    import shutil
    def passthrough(msg):
        print(f"   {msg} -> vocals passed through unpatched")
        shutil.copyfile(in_mid, out_mid)
    pm = pretty_midi.PrettyMIDI(in_mid)
    vnotes = sorted((n.start, n.end, n.pitch) for n in pm.instruments[0].notes)
    _t, _tempi = pm.get_tempo_changes()
    bpm = float(_tempi[0]) if len(_tempi) else 120.0
    gate = (60.0 / bpm) * VGAP_GATE_BEATS
    try:
        notes = run_basic_pitch(vocals_wav, bp_cache_v)
        try:
            sal = note_salience(vocals_wav, notes)
        except Exception as ex:
            print(f"   loudness measurement failed ({ex}) -> basic-pitch amps only")
            sal = None
        line = dp_melody(notes, sal)
    except Exception as ex:
        return passthrough(f"vocal-patch transcription failed ({ex})")
    regions, swapped = [], []
    if crepe_npz and os.path.exists(crepe_npz):
        try:
            cz = np.load(crepe_npz)
            tk = next((k for k in cz.files if k.lower() in ('t', 'time', 'times')), None)
            ck = next((k for k in cz.files if k.lower().startswith('conf')), None)
            if tk and ck:
                regions = _low_conf_regions(cz[tk], cz[ck], bpm)
            else:
                print(f"   pitch cache keys {list(cz.files)}: no time/confidence pair "
                      f"-> confidence scan skipped")
        except Exception as ex:
            print(f"   confidence scan skipped ({ex})")
    regions = _merge_spans(list(regions))

    is_vocal = vocal_presence(vocals_wav, other_wav)
    if is_vocal is not None and regions:
        sung = [r for r in regions if is_vocal(*r)]
        for (r0, r1) in [r for r in regions if r not in sung]:
            print(f"   {r0:.1f}-{r1:.1f}s: no vocal present -> left to the instrumental fill")
        regions = sung

    vnotes, swapped = _replace_regions(vnotes, line, regions)
    for (r0, r1, rem, ins) in swapped:
        print(f"   {r0:.1f}-{r1:.1f}s: replaced {rem} note(s) with {ins} from vocals.wav")
    swapped_starts = {round(a, 6) for a, _, _, _ in swapped}
    thinned = []
    for (r0, r1) in regions:
        if round(r0, 6) in swapped_starts: continue
        vnotes, n_in, n_keep = _thin_region(vnotes, r0, r1, 60.0 / bpm)
        if n_in > n_keep:
            thinned.append((r0, r1))
            print(f"   {r0:.1f}-{r1:.1f}s: no line to substitute -> thinned "
                  f"{n_in} note(s) to {n_keep}")

    song_end = line[-1][1] if line else (vnotes[-1][1] if vnotes else 0.0)
    holes, added, skipped = _patch_holes(vnotes, line, gate, song_end, is_vocal)
    if holes:
        print("   vocal gap(s) at " + ", ".join(f"{h0:.1f}-{h1:.1f}s" for h0, h1 in holes))
    if skipped:
        print("   instrumental gap(s) left for stage 7: "
              + ", ".join(f"{h0:.1f}-{h1:.1f}s" for h0, h1 in skipped))
    if not added and not swapped and not thinned:
        print(f"   {len(holes)} vocal gap(s) >= {VGAP_GATE_BEATS:.0f} beats, "
              f"{len(regions)} unreliable region(s) | nothing to change")
        shutil.copyfile(in_mid, out_mid); return
    out_pm = pretty_midi.PrettyMIDI(initial_tempo=int(round(bpm)))
    out_pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    inst = pretty_midi.Instrument(program=0, name="Vocals (patched)")
    for (s, e, p) in sorted(vnotes + added):
        inst.notes.append(pretty_midi.Note(velocity=85, pitch=int(max(0, min(127, p))),
                                           start=float(s), end=float(e)))
    out_pm.instruments.append(inst)
    out_pm.write(out_mid)
    big = max((h1 - h0 for h0, h1 in holes), default=0.0)
    print(f"   {len(swapped)} region(s) replaced | {len(holes)} vocal gap(s)"
          f"{f' (longest {big:.1f}s)' if holes else ''} | +{len(added)} note(s) -> {out_mid}")

# stage 7 instrumental fill
def run_rest_fill(in_mid, other_wav, bp_cache, out_mid, dp_mid=None):
    import shutil
    def passthrough(msg):
        print(f"   {msg} -> vocals passed through unfilled")
        shutil.copyfile(in_mid, out_mid)

    pm = pretty_midi.PrettyMIDI(in_mid)
    vnotes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
    _t, _tempi = pm.get_tempo_changes()
    bpm = float(_tempi[0]) if len(_tempi) else 120.0
    gate = (60.0 / bpm) * FILL_GATE_BEATS

    rests = []
    if vnotes and vnotes[0].start >= gate:
        rests.append((0.0, vnotes[0].start))
    for i in range(len(vnotes) - 1):
        g0, g1 = vnotes[i].end, vnotes[i + 1].start
        if g1 - g0 >= gate:
            rests.append((g0, g1))
    if not rests:
        return passthrough(f"no vocal rests >= {FILL_GATE_BEATS:.0f} beats")

    lead_wav = os.path.join(os.path.dirname(other_wav), "lead.wav")
    if os.path.isfile(lead_wav):
        print("   found lead.wav -> using it INSTEAD of other.wav")
        other_wav = lead_wav
        bp_cache = bp_cache[:-4] + "_lead.npz"
    try:
        notes = run_basic_pitch(other_wav, bp_cache)
        try:
            sal = note_salience(other_wav, notes)
        except Exception as ex:
            print(f"   loudness measurement failed ({ex}) -> basic-pitch amps only")
            sal = None
        line = dp_melody(notes, sal)
    except Exception as ex:
        return passthrough(f"instrumental transcription failed ({ex})")
    if dp_mid and line:
        dpm = pretty_midi.PrettyMIDI(initial_tempo=int(round(bpm)))
        dpm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
        di = pretty_midi.Instrument(program=0, name="Instrumental melody")
        for (s, e, p) in line:
            di.notes.append(pretty_midi.Note(velocity=85, pitch=int(max(0, min(127, p))),
                                             start=float(s), end=float(e)))
        dpm.instruments.append(di)
        dpm.write(dp_mid)
        print(f"   full-song instrumental line -> {dp_mid}")
    if line:
        tail0, tail1 = vnotes[-1].end, line[-1][1]
        if tail1 - tail0 >= gate:
            rests.append((tail0, tail1))

    inst = pm.instruments[0]
    added = filled = 0
    for (a, b) in rests:
        seg = [(s, min(e, b), p) for (s, e, p) in line if a <= s < b]
        seg = [(s, e, p) for (s, e, p) in seg if e - s >= 0.08]
        if len(seg) < FILL_MIN_NOTES:
            continue
        for (s, e, p) in seg:
            inst.notes.append(pretty_midi.Note(velocity=85, pitch=int(max(0, min(127, p))),
                                               start=float(s), end=float(e)))
        added += len(seg); filled += 1
    pm.write(out_mid)
    print(f"   {len(rests)} gated rest(s) | filled {filled} with {added} melody notes -> {out_mid}")


# left hand patterns
BLOCK_VARIANTS = {"inv1", "inv2", "open5", "oct158"}

def acc_block_pitches(name, r, t, f):
    if name == "inv1":   return [t.transpose(-12), f.transpose(-12), r]
    if name == "inv2":   return [f.transpose(-12), r, t]
    if name == "open5":  return [r, f]
    if name == "oct158": return [r.transpose(-12), f.transpose(-12), r]
    return [r, t, f]

def _tile4(template, beats):
    out, base = [], 0.0
    while base < beats - 1e-9:
        for rel, ql, ps in template:
            pos = base + rel
            if pos >= beats - 1e-9: break
            out.append((pos, min(ql, beats - pos), ps))
        base += 4.0
    return out

def acc_build(pat, beats, r, t, f):
    nb = int(round(beats))
    r_lo, f_lo = r.transpose(-12), f.transpose(-12)
    if pat == "pulse":
        return [(float(i), 1.0, [r]) for i in range(nb)]
    if pat == "asc_ninth":
        ninth = r.transpose(2)
        return _tile4([(0.0, 0.5, [r_lo]), (0.5, 0.5, [f_lo]), (1.0, 0.5, [r]),
                       (1.5, 0.5, [ninth]), (2.0, 2.0, [t])], beats)
    if pat == "asc_twelfth":
        return _tile4([(0.0, 0.5, [r_lo]), (0.5, 0.5, [f_lo]), (1.0, 0.5, [r]),
                       (1.5, 0.5, [t]), (2.0, 2.0, [f])], beats)
    if pat == "climb":
        cell = [r_lo, f_lo, r, f_lo, t, f_lo, r, f_lo]
        return [(i*0.5, 0.5, [cell[i % len(cell)]]) for i in range(int(round(beats*2)))]
    if pat == "updown":
        cell = [r_lo, f_lo, r, t, r, f_lo, r_lo, f_lo]
        return [(i*0.5, 0.5, [cell[i % len(cell)]]) for i in range(int(round(beats*2)))]
    if pat == "pump":
        return [(i*0.5, 0.5, [r_lo] if i % 2 == 0 else [f_lo, r])
                for i in range(int(round(beats*2)))]
    return [(0.0, float(beats), [r, t, f])]


# stage 10 left hand rests
def run_bass_silence(xml_in, xml_out, bass_wav, other_wav):
    score = m21.converter.parse(xml_in)
    tm = score.flatten().getElementsByClass(m21.tempo.MetronomeMark)
    bpm = float(tm[0].number) if tm else 128.0
    spb = 60.0 / bpm
    lh = next((p for p in score.parts if (p.partName or "").lower().startswith("left")), score.parts[1])

    yb, _ = librosa.load(bass_wav,  sr=SR, mono=True)
    yo, _ = librosa.load(other_wav, sr=SR, mono=True)
    rb = librosa.feature.rms(y=yb, frame_length=640, hop_length=HOP)[0]
    ro = librosa.feature.rms(y=yo, frame_length=640, hop_length=HOP)[0]
    tb = librosa.frames_to_time(np.arange(len(rb)), sr=SR, hop_length=HOP)
    to = librosa.frames_to_time(np.arange(len(ro)), sr=SR, hop_length=HOP)
    def emed(t, r, a, b):
        m = (t >= a) & (t < b)
        return float(np.median(r[m])) if m.any() else 0.0

    items = []
    for m in lh.getElementsByClass('Measure'):
        for el in m.notes:
            s_ql = float(m.offset) + float(el.offset)
            e_ql = s_ql + float(el.quarterLength)
            a, b = s_ql * spb, e_ql * spb
            items.append({'m': m, 'el': el, 'off': float(el.offset), 'ql': float(el.quarterLength),
                          'meas': m.number, 's_ql': s_ql, 'e_ql': e_ql, 'a': a, 'b': b,
                          'be': emed(tb, rb, a, b), 'oe': emed(to, ro, a, b)})

    bass_floor  = float(np.median([x['be'] for x in items])) * SIL_FLOOR_FRAC
    other_floor = float(np.median([x['oe'] for x in items])) * SIL_FLOOR_FRAC
    for it in items:
        it['dead'] = it['be'] < bass_floor and it['oe'] < other_floor

    runs, cur = [], []
    for it in items:
        if not it['dead']:
            if cur: runs.append(cur); cur = []
            continue
        if cur and abs(it['s_ql'] - cur[-1]['e_ql']) < 1e-6:
            cur.append(it)
        else:
            if cur: runs.append(cur)
            cur = [it]
    if cur: runs.append(cur)

    silenced = []
    for run in runs:
        total = sum(x['ql'] for x in run)
        if total >= SIL_GATE_BEATS:
            for it in run:
                it['m'].remove(it['el'])
                it['m'].insert(it['off'], m21.note.Rest(quarterLength=it['ql']))
                it['sil'] = True
                silenced.append(it)

    blips = 0
    live_runs, cur = [], []
    for it in items:
        if it.get('sil'):
            if cur: live_runs.append(cur); cur = []
            continue
        if cur and abs(it['s_ql'] - cur[-1]['e_ql']) < 1e-6:
            cur.append(it)
        else:
            if cur: live_runs.append(cur)
            cur = [it]
    if cur: live_runs.append(cur)
    for run in live_runs:
        if sum(x['ql'] for x in run) <= 1.0 + 1e-6:
            for it in run:
                it['m'].remove(it['el'])
                it['m'].insert(it['off'], m21.note.Rest(quarterLength=it['ql']))
                it['sil'] = True
                blips += 1

    els = list(lh.flatten().notesAndRests)
    for i, el in enumerate(els):
        if el.isRest or el.tie is None:
            continue
        prv = els[i-1] if i > 0 else None
        nxt = els[i+1] if i+1 < len(els) else None
        bad_fwd = el.tie.type in ('start', 'continue') and (nxt is None or nxt.isRest)
        bad_bwd = el.tie.type in ('stop', 'continue') and (prv is None or prv.isRest)
        if bad_fwd and bad_bwd:        el.tie = None
        elif bad_fwd and el.tie.type == 'continue': el.tie = m21.tie.Tie('stop')
        elif bad_fwd:                  el.tie = None
        elif bad_bwd and el.tie.type == 'continue': el.tie = m21.tie.Tie('start')
        elif bad_bwd:                  el.tie = None

    probs = measure_health(score)
    os.makedirs(os.path.dirname(xml_out) or ".", exist_ok=True)
    score.write('musicxml', fp=xml_out)
    print(f"   silenced {len(silenced)} block(s) + {blips} islanded blip(s) "
          f"| bad measures {len(probs)} -> {xml_out}")
    for pn, mn, why in probs[:6]:
        print(f"   !! {pn} measure {mn}: {why}")


# stage 11 left hand texture
def run_accompaniment_variation(xml_in, xml_out, drums_wav, rng=None):
    score = m21.converter.parse(xml_in)
    tm = score.flatten().getElementsByClass(m21.tempo.MetronomeMark)
    bpm = float(tm[0].number) if tm else 128.0
    spb = 60.0 / bpm
    lh = next((p for p in score.parts if (p.partName or "").lower().startswith("left")), score.parts[1])
    lh_measures = list(lh.getElementsByClass('Measure'))
    starts = [float(mm.offset) for mm in lh_measures]
    def measure_for(abs_off):
        i = max(0, min(bisect.bisect_right(starts, abs_off + 1e-6) - 1, len(lh_measures) - 1))
        return lh_measures[i], starts[i]

    ks = None
    for m in lh_measures:
        f = m.getElementsByClass(m21.key.KeySignature)
        if len(f): ks = f[0]; break
    if ks is None:
        f = score.flatten().getElementsByClass(m21.key.KeySignature)
        ks = f[0] if len(f) else m21.key.KeySignature(6)

    yd, _ = librosa.load(drums_wav, sr=SR, mono=True)
    onset_env = librosa.onset.onset_strength(y=yd, sr=SR, hop_length=HOP)
    ot = librosa.times_like(onset_env, sr=SR, hop_length=HOP)
    def activity(a, b):
        m = (ot >= a) & (ot < b)
        return float(np.mean(onset_env[m])) if m.any() else 0.0

    blocks = []
    for m in lh_measures:
        for el in m.notes:
            s = float(m.offset) + float(el.offset)
            pit = sorted((el.pitches if el.isChord else [el.pitch]), key=lambda p: p.midi)
            blocks.append({'m': m, 'el': el, 's': s, 'e': s + float(el.quarterLength),
                           'pset': frozenset(p.midi for p in pit), 'pitches': pit})
    blocks.sort(key=lambda b: b['s'])
    logicals = []
    for b in blocks:
        if logicals and logicals[-1]['pset'] == b['pset'] and abs(logicals[-1]['e'] - b['s']) < 1e-6:
            logicals[-1]['e'] = b['e']; logicals[-1]['parts'].append(b)
        else:
            logicals.append({'pset': b['pset'], 's': b['s'], 'e': b['e'], 'pitches': b['pitches'], 'parts': [b]})

    for lc in logicals:
        lc['beats'] = lc['e'] - lc['s']
        lc['raw'] = activity(lc['s'] * spb, lc['e'] * spb)
    for lc in logicals:
        mid = (lc['s'] + lc['e']) / 2 * spb
        near = [x['raw'] for x in logicals if abs((x['s']+x['e'])/2*spb - mid) <= ACC_SMOOTH_SEC]
        lc['smooth'] = float(np.mean(near)) if near else lc['raw']
    sm = np.array([lc['smooth'] for lc in logicals])
    t1, t2 = np.percentile(sm, 33), np.percentile(sm, 66)
    def tier_of(v): return "calm" if v < t1 else ("medium" if v < t2 else "busy")
    for lc in logicals:
        lc['tier'] = tier_of(lc['smooth'])

    merged = []
    for lc in logicals:
        if (round(lc['beats']) == 1 and lc['tier'] == "busy"
                and merged and round(merged[-1]['beats']) >= 2
                and abs(merged[-1]['e'] - lc['s']) < 1e-6):
            prev = merged[-1]
            prev['e'] = lc['e']; prev['beats'] = prev['e'] - prev['s']
            prev['parts'].extend(lc['parts'])
        else:
            merged.append(lc)
    logicals = merged

    def _reharmonize(parts, pitches):
        for part in parts:
            el = part['el']; meas = part['m']; off = float(el.offset)
            nc = m21.chord.Chord([copy.deepcopy(pp) for pp in pitches])
            nc.quarterLength = el.quarterLength
            if el.tie is not None: nc.tie = m21.tie.Tie(el.tie.type)
            meas.remove(el); meas.insert(off, nc)
            part['el'] = nc
    absorbed = 0
    res = []
    for k, lc in enumerate(logicals):
        prev = res[-1] if res else None
        nxt = logicals[k + 1] if k + 1 < len(logicals) else None
        cont_prev = prev is not None and abs(prev['e'] - lc['s']) < 1e-6
        cont_next = nxt is not None and abs(lc['e'] - nxt['s']) < 1e-6
        if lc['beats'] <= 1.0 + 1e-6:
            if (cont_prev and lc['pset'] != prev['pset']
                    and (nxt is None or lc['pset'] != nxt['pset'])
                    and len(prev['pitches']) == 3):
                _reharmonize(lc['parts'], prev['pitches'])
                prev['e'] = lc['e']; prev['beats'] = prev['e'] - prev['s']
                prev['parts'].extend(lc['parts'])
                absorbed += 1
                continue
            if ((not cont_prev) and cont_next and lc['pset'] != nxt['pset']
                    and len(nxt['pitches']) == 3):
                _reharmonize(lc['parts'], nxt['pitches'])
                nxt['s'] = lc['s']; nxt['beats'] = nxt['e'] - nxt['s']
                nxt['parts'] = lc['parts'] + nxt['parts']
                absorbed += 1
                continue
        res.append(lc)
    logicals = res

    for k, lc in enumerate(logicals):
        p = logicals[k - 1] if k > 0 else None
        n = logicals[k + 1] if k + 1 < len(logicals) else None
        lc['kin'] = ((p is not None and lc['pset'] == p['pset']) or
                     (n is not None and lc['pset'] == n['pset']))

    i = 0
    while i < len(logicals):
        j = i
        while j + 1 < len(logicals) and logicals[j + 1]['tier'] == logicals[i]['tier']:
            j += 1
        tier = logicals[i]['tier']
        menu = {"medium": MEDIUM_MENU, "busy": BUSY_MENU, "calm": CALM_MENU}[tier]
        choice = menu[0] if rng is None else menu[int(rng.integers(len(menu)))]
        for k in range(i, j + 1):
            logicals[k]['pat_choice'] = choice
        i = j + 1

    def decide(beats, tier, ntones, choice, kin=False):
        if ntones != 3: return "block"
        if tier == "calm": return choice
        if tier == "medium": return choice
        if round(beats) <= 1 and not kin: return "block"
        return choice

    def place_note(plist, abs_off, ql):
        cursor, remaining, segs = abs_off, ql, []
        while remaining > 1e-9:
            meas, mstart = measure_for(cursor)
            seg = min(remaining, (mstart + 4.0) - cursor)
            if seg <= 1e-9: break
            segs.append((cursor, seg, meas, mstart)); cursor += seg; remaining -= seg
        for i, (off, seg, meas, mstart) in enumerate(segs):
            pc = [copy.deepcopy(p) for p in plist]
            new = m21.note.Note(pc[0]) if len(pc) == 1 else m21.chord.Chord(pc)
            new.quarterLength = seg
            if len(segs) > 1:
                new.tie = m21.tie.Tie('start' if i == 0 else ('stop' if i == len(segs)-1 else 'continue'))
            meas.insert(off - mstart, new)

    def place_held(pitches, abs_start, dur):
        cursor, remaining, segs = abs_start, dur, []
        while remaining > 1e-9:
            meas, mstart = measure_for(cursor)
            seg = min(remaining, (mstart + 4.0) - cursor)
            if seg <= 1e-9: break
            segs.append((cursor, seg, meas, mstart)); cursor += seg; remaining -= seg
        for i, (off, seg, meas, mstart) in enumerate(segs):
            ch = m21.chord.Chord([copy.deepcopy(p) for p in pitches]); ch.quarterLength = seg
            if len(segs) > 1:
                ch.tie = m21.tie.Tie('start' if i == 0 else ('stop' if i == len(segs)-1 else 'continue'))
            meas.insert(off - mstart, ch)

    counts, one_beat_blocks = {}, 0
    last_lc = logicals[-1] if logicals else None
    for lc in logicals:
        pat = decide(lc['beats'], lc['tier'], len(lc['pitches']),
                     lc.get('pat_choice', 'block'), lc.get('kin', False))
        r, t, f = lc['pitches'][0], lc['pitches'][1], lc['pitches'][2]

        if lc is last_lc and pat != "block" and pat not in BLOCK_VARIANTS:
            counts['resolve'] = counts.get('resolve', 0) + 1
            for part in lc['parts']: part['m'].remove(part['el'])
            D = lc['beats']; block_beats = min(4.0, D); lead = D - block_beats
            if lead > 0:
                for rel, ql, plist in acc_build(pat, lead, r, t, f):
                    place_note(plist, lc['s'] + rel, ql)
            place_held([r, t, f], lc['s'] + lead, block_beats)
            continue

        counts[pat] = counts.get(pat, 0) + 1
        if pat == "block":
            if round(lc['beats']) == 1: one_beat_blocks += 1
            continue
        if pat in BLOCK_VARIANTS:
            vp = acc_block_pitches(pat, r, t, f)
            for part in lc['parts']:
                el = part['el']; meas = part['m']; off = float(el.offset)
                nc = m21.chord.Chord([copy.deepcopy(p) for p in vp])
                nc.quarterLength = el.quarterLength
                if el.tie is not None: nc.tie = m21.tie.Tie(el.tie.type)
                meas.remove(el); meas.insert(off, nc)
            continue
        for part in lc['parts']: part['m'].remove(part['el'])
        for rel, ql, plist in acc_build(pat, lc['beats'], r, t, f):
            place_note(plist, lc['s'] + rel, ql)
    def pset(e): return None if (e is None or e.isRest) else frozenset(p.midi for p in (e.pitches if e.isChord else [e.pitch]))
    els = list(lh.flatten().notesAndRests)
    for i, el in enumerate(els):
        if el.isRest or el.tie is None: continue
        tt = el.tie.type
        prv = els[i-1] if i > 0 else None
        nxt = els[i+1] if i+1 < len(els) else None
        fwd_ok = (nxt is not None and not nxt.isRest and nxt.tie is not None
                  and nxt.tie.type in ('stop', 'continue') and pset(nxt) == pset(el))
        bwd_ok = (prv is not None and not prv.isRest and prv.tie is not None
                  and prv.tie.type in ('start', 'continue') and pset(prv) == pset(el))
        need_fwd, need_bwd = tt in ('start', 'continue'), tt in ('stop', 'continue')
        nf, nb = (need_fwd and fwd_ok), (need_bwd and bwd_ok)
        if need_fwd and need_bwd:
            el.tie = None if not (nf or nb) else (m21.tie.Tie('start') if nf and not nb else
                                                  (m21.tie.Tie('stop') if nb and not nf else el.tie))
        elif need_fwd: el.tie = el.tie if nf else None
        elif need_bwd: el.tie = el.tie if nb else None

    try: key = ks.asKey('major')
    except Exception: key = m21.key.KeySignature(6).asKey('major')
    pc_map = {}
    for p in key.getPitches(): pc_map[p.pitchClass] = p.name
    prefer_sharp = (key.sharps or 0) >= 0
    for pc in range(12):
        if pc in pc_map: continue
        pp = m21.pitch.Pitch(midi=60 + pc)
        if prefer_sharp and pp.accidental and pp.accidental.alter < 0: pp = pp.getEnharmonic()
        elif (not prefer_sharp) and pp.accidental and pp.accidental.alter > 0: pp = pp.getEnharmonic()
        pc_map[pc] = pp.name
    def respell(pp):
        out = m21.pitch.Pitch(pc_map[pp.midi % 12]); out.octave = pp.octave
        if out.midi != pp.midi: out.octave += round((pp.midi - out.midi) / 12)
        return out
    keymap = {p.step: p.accidental.alter for p in ks.alteredPitches}
    for m in lh_measures:
        state = {}
        for el in sorted(m.notes, key=lambda x: x.offset):
            for pp in (el.pitches if el.isChord else [el.pitch]):
                tt = respell(pp); pp.name, pp.octave = tt.name, tt.octave
                ka = keymap.get(pp.step, 0.0)
                na = pp.accidental.alter if pp.accidental else 0.0
                so = (pp.step, pp.octave); exp = state.get(so, ka)
                if na != exp:
                    if pp.accidental is None: pp.accidental = m21.pitch.Accidental('natural')
                    pp.accidental.displayStatus = True; state[so] = na
                elif pp.accidental is not None:
                    pp.accidental.displayStatus = False

    for m in lh_measures:
        try: m.makeBeams(inPlace=True)
        except Exception: pass

    probs = measure_health(score)
    os.makedirs(os.path.dirname(xml_out) or ".", exist_ok=True)
    score.write('musicxml', fp=xml_out)
    print(f"   patterns {dict(sorted(counts.items()))} | absorbed {absorbed} orphan(s) | 1-beat blocks left {one_beat_blocks} "
          f"| bad measures {len(probs)} -> {xml_out}")
    for pn, mn, why in probs[:6]:
        print(f"   !! {pn} measure {mn}: {why}")


# title and composer
def set_score_info(xml_path, title=None, composer=None):
    import re
    from xml.sax.saxutils import escape
    with open(xml_path, encoding="utf-8") as fh:
        xml = fh.read()
    cut = xml.find("<part-list")
    head, body = (xml[:cut], xml[cut:]) if cut > 0 else (xml, "")

    title = (title or "").strip()
    if title:
        line = f"<movement-title>{escape(title)}</movement-title>"
        m = re.search(r"([ \t]*)<movement-title>.*?</movement-title>\n?", head, re.S)
        if m:
            head = head[:m.start()] + m.group(1) + line + "\n" + head[m.end():]
        else:
            head = head.replace("<identification>", line + "\n  <identification>", 1)

    composer = (composer or "").strip()
    line = f'<creator type="composer">{escape(composer)}</creator>'
    m = re.search(r'([ \t]*)<creator type="composer">.*?</creator>\n?', head, re.S)
    if m:
        head = head[:m.start()] + (m.group(1) + line + "\n" if composer else "") + head[m.end():]
    elif composer:
        head = head.replace("<identification>", "<identification>\n    " + line, 1)

    with open(xml_path, "w", encoding="utf-8") as fh:
        fh.write(head + body)
    return xml_path


# run all stages
def migrate_old_filenames(out_dir, song):
    if not os.path.isdir(out_dir): return
    def new_name(f):
        if not f.startswith(song): return None
        if f.endswith("_beats.npz"):      return "2" + f
        if f.endswith("_crepe.npz"):      return "3" + f
        for pat in ("_vocals.mid", "_vocals_extended.mid", "_vocals_patched.mid", "_vocals_filled.mid"):
            if f.endswith(pat):           return "4" + f
        if f.endswith("_basicpitch.npz") or f.endswith("_basicpitch_lead.npz"):
            return "5" + f
        if f.endswith("_other_dp.mid"):   return "5" + f
        if f.endswith("_chords.mid") or f.endswith("_chords.txt"): return "6" + f
        if "_base" in f and (f.endswith(".musicxml") or f.endswith(".musicxml.ver")):
            return "7" + f
        if "_silenced" in f and f.endswith(".musicxml"): return "8" + f
        if f.endswith(".musicxml"):       return "FINAL_" + f
        return None
    moved = 0
    for f in sorted(os.listdir(out_dir)):
        nn = new_name(f)
        if nn is None: continue
        dst = os.path.join(out_dir, nn)
        if os.path.exists(dst): continue
        os.rename(os.path.join(out_dir, f), dst); moved += 1
        print(f"   renamed {f} -> {nn}")
    if moved:
        print(f"   ({moved} file(s) migrated to the numbered scheme; caches are kept, "
              f"so nothing is rebuilt)")


def run_pipeline(input_audio, force=False, seed=None, title=None, composer=None):
    print("=" * 72)
    print(f"UNIFIED PIANO TRANSCRIPTION PIPELINE   ({os.path.basename(input_audio)})"
          + (f"   [seed {seed}]" if seed is not None else ""))
    print("=" * 72)
    rng_gs  = np.random.default_rng([int(seed), 0]) if seed is not None else None
    rng_acc = np.random.default_rng([int(seed), 1]) if seed is not None else None
    tag = f"_seed{seed}" if seed is not None else ""

    print("\n[0/11] Stem separation")
    stem_dir, song = run_demucs(input_audio)

    out_dir = os.path.join(OUT_ROOT, song)
    os.makedirs(out_dir, exist_ok=True)
    migrate_old_filenames(out_dir, song)
    P = lambda name: os.path.join(out_dir, name)

    vocals_wav = os.path.join(stem_dir, "vocals.wav")
    bass_wav   = os.path.join(stem_dir, "bass.wav")
    drums_wav  = os.path.join(stem_dir, "drums.wav")
    other_wav  = os.path.join(stem_dir, "other.wav")
    accomp_wav = os.path.join(stem_dir, "accompaniment.wav")

    beats_npz      = P(f"2{song}_beats.npz")
    crepe_npz      = P(f"3{song}_crepe.npz")
    vocals_mid     = P(f"4{song}_vocals.mid")
    vocals_ext_mid = P(f"4{song}_vocals_extended.mid")
    vocals_fill_mid = P(f"4{song}_vocals_filled.mid")
    vocals_patch_mid = P(f"4{song}_vocals_patched.mid")
    bp_cache       = P(f"5{song}_basicpitch.npz")
    bp_vocals_cache = P(f"5{song}_basicpitch_vocals.npz")
    other_dp_mid   = P(f"5{song}_other_dp.mid")
    chords_mid     = P(f"6{song}_chords.mid")
    chords_txt     = P(f"6{song}_chords.txt")
    base_xml       = P(f"7{song}_base{tag}.musicxml")
    silenced_xml   = P(f"8{song}_silenced{tag}.musicxml")
    final_xml      = P(f"FINAL_{song}{tag}.musicxml")
    def have(*paths): return all(os.path.exists(p) for p in paths)

    def beats_current(p):
        try:
            c = np.load(p, allow_pickle=True)
            return "repair" in c.files and int(c["repair"]) == BEAT_REPAIR_VERSION
        except Exception:
            return False

    def gs_current(p):
        try:
            first = open(p + ".ver").readline()
            return "".join(ch for ch in first if ch.isdigit()) == str(GS_VERSION)
        except OSError: return False

    if force:
        for f in (crepe_npz, beats_npz):
            if os.path.exists(f): os.remove(f)

    do_voc  = force or not have(vocals_mid)
    def mel_current(p):
        try:
            first = open(p + ".ver").readline()
            return "".join(ch for ch in first if ch.isdigit()) == str(MEL_VERSION)
        except OSError: return False
    do_patch = (do_voc or not have(vocals_patch_mid)
                or not mel_current(vocals_patch_mid))
    do_ext  = do_patch or not have(vocals_ext_mid)
    do_fill = ENABLE_REST_FILL and (do_ext or not have(vocals_fill_mid))
    melody_mid = vocals_fill_mid if ENABLE_REST_FILL else vocals_ext_mid
    up_mel  = do_fill if ENABLE_REST_FILL else do_ext
    grid_stale = have(beats_npz) and not beats_current(beats_npz)
    do_chd  = force or not have(chords_mid) or grid_stale
    do_base = up_mel or do_chd or not have(base_xml) or not gs_current(base_xml)
    do_sil  = do_base or not have(silenced_xml)
    do_var  = do_sil or not have(final_xml)
    need_tempo = do_voc or do_chd or do_base
    if grid_stale:
        print("\n   (cached beat grid predates the current repair -> "
              "chords and score will rebuild)")
    tempo = beat_times = None

    print("\n[1/11] Accompaniment mixdown")
    if need_tempo and not have(accomp_wav):
        run_accompaniment(stem_dir, accomp_wav)
    elif have(accomp_wav):
        print("   present -> reuse")
    else:
        print("   not needed (vocals + chords already cached)")

    print("\n[2/11] Tempo and beat grid")
    if need_tempo:
        tempo, beat_times, grid_changed = detect_tempo_and_beats(accomp_wav, beats_npz)
        if grid_changed:
            print("   the grid moved -> chords and score will be rebuilt on it")
            do_chd = do_base = do_sil = do_var = True
    else:
        print("   not needed (vocals + chords already cached)")

    print("\n[3-4/11] Vocal transcription")
    if do_voc: run_vocals(vocals_wav, crepe_npz, vocals_mid, tempo)
    else:      print(f"   reuse {vocals_mid}")

    print("\n[5/11] Vocal recovery (unresolved notes from vocals.wav)")
    if do_patch:
        run_vocal_patch(vocals_mid, vocals_wav, bp_vocals_cache, vocals_patch_mid,
                        crepe_npz=crepe_npz, other_wav=other_wav)
        with open(vocals_patch_mid + ".ver", "w") as fh:
            fh.write(f"melody-chain version: {MEL_VERSION}\n\n"
                     f"Records which version of the stage 5-7 melody chain produced\n"
                     f"this file. If it does not match MEL_VERSION in pipeline.py, the\n"
                     f"chain is rebuilt automatically. Safe to delete; that forces one\n"
                     f"rebuild.\n")
    else:
        print(f"   (cached) {vocals_patch_mid}")

    print("\n[6/11] Sustained note tails")
    if do_ext: run_extend(vocals_patch_mid, vocals_wav, vocals_ext_mid)
    else:      print(f"   reuse {vocals_ext_mid}")

    print("\n[7/11] Instrumental fill (other.wav melody into vocal rests)")
    if not ENABLE_REST_FILL:
        print("   disabled (ENABLE_REST_FILL=False) -> the extended vocal line is used")
    elif do_fill:
        run_rest_fill(vocals_ext_mid, other_wav, bp_cache, vocals_fill_mid, dp_mid=other_dp_mid)
    else:
        print(f"   reuse {vocals_fill_mid}")

    print("\n[8/11] Chord extraction")
    if do_chd: run_chords(bass_wav, other_wav, accomp_wav, chords_mid, chords_txt, tempo, beat_times)
    else:      print(f"   reuse {chords_mid}")

    print("\n[9/11] Grand staff assembly")
    if do_base:
        _, keyname, bpm, bad, _ = grand_staff_make(melody_mid, chords_mid, base_xml,
                                                   rng=rng_gs, beat_times=beat_times,
                                                   tempo=tempo)
        with open(base_xml + ".ver", "w") as fh:
            fh.write(f"grand-staff assembly version: {GS_VERSION}\n"
                     f"\n"
                     f"Records which version of the grand-staff assembly in pipeline.py\n"
                     f"produced the score beside it. If it does not match GS_VERSION,\n"
                     f"that score is stale and is rebuilt automatically on the next run,\n"
                     f"so a change to the assembly always reaches the page. Safe to\n"
                     f"delete; that forces one rebuild.\n")
        print(f"   key {keyname} | tempo {bpm:.0f} BPM | bad measures {bad} -> {base_xml}")
    else:
        print(f"   reuse {base_xml}")

    print("\n[10/11] Left-hand rests")
    if do_sil: run_bass_silence(base_xml, silenced_xml, bass_wav, other_wav)
    else:      print(f"   reuse {silenced_xml}")

    print("\n[11/11] Accompaniment texture")
    if do_var: run_accompaniment_variation(silenced_xml, final_xml, drums_wav, rng=rng_acc)
    else:      print(f"   reuse {final_xml}")

    sheet_title = (title or "").strip() or song
    set_score_info(final_xml, sheet_title, composer)

    print("\n" + "=" * 72)
    if not (do_voc or do_ext or do_patch or do_fill or do_chd or do_base or do_sil or do_var):
        print("Everything was already cached; nothing needed rebuilding.")
    print(f"DONE  ->  {final_xml}")
    print(f'Titled "{sheet_title}"' + (f' by {composer.strip()}' if (composer or "").strip() else ""))
    print("Open the file above. The numbered files beside it are the intermediate")
    print("stages in build order; 7*_base and 8*_silenced are earlier states of the")
    print("same score, not the finished one.")
    print("=" * 72)
    return final_xml


def prompt_for_audio():
    d = os.path.join("data", "input")
    while True:
        name = input("\nAudio file to transcribe (e.g. song.mp3): ").strip().strip('"').strip("'")
        if not name:
            continue
        for cand in (name, os.path.join(d, name)):
            if os.path.isfile(cand):
                return cand
        song = os.path.splitext(os.path.basename(name))[0]
        if os.path.isfile(os.path.join(STEM_ROOT, DEMUCS_MODEL, song, "vocals.wav")):
            print(f"   (audio not found, but stems already exist for '{song}' -> using those)")
            return name
        print(f"   '{name}' not found.")
        if os.path.isdir(d):
            files = sorted(f for f in os.listdir(d) if not f.startswith('.'))
            if files:
                print("   available in data/input/:")
                for f in files: print("      -", f)
        print("   (type a name or full path, or press Ctrl-C to quit)")


def song_has_cached_products(audio):
    song = os.path.splitext(os.path.basename(audio))[0]
    out_dir = os.path.join(OUT_ROOT, song)
    if os.path.isdir(out_dir) and any(not f.startswith('.') for f in os.listdir(out_dir)):
        return True
    return os.path.isfile(os.path.join(STEM_ROOT, DEMUCS_MODEL, song, "vocals.wav"))


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else prompt_for_audio()
    song_name = os.path.splitext(os.path.basename(audio))[0]
    title = sys.argv[3] if len(sys.argv) > 3 else \
        input(f"Title for the sheet (Enter = {song_name}): ").strip()
    composer = sys.argv[4] if len(sys.argv) > 4 else \
        input("Composer (Enter = leave it blank): ").strip()
    force = FORCE_REBUILD
    seed = None
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])
    else:
        s = input("Seed for a variant sheet (Enter = default | number | r = random): ").strip().lower()
        if s == "r":
            seed = int(np.random.default_rng().integers(1, 100000))
            print(f"   random seed: {seed}")
        elif s:
            try: seed = int(s)
            except ValueError: print("   (not a number -> running default)")
    if not force and song_has_cached_products(audio):
        ans = input("Regenerate everything from scratch, including CREPE? [y/N]: ").strip().lower()
        force = ans in ("y", "yes")
    run_pipeline(audio, force=force, seed=seed, title=title, composer=composer)
