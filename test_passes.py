#!/usr/bin/env python3
"""
test_passes.py -- regression tests for the note-level logic in pipeline.py

Covers the passes that decide what ends up on the page (merging, quantization,
bar decluttering, register repair), the gap handling in stage 5, the beat grid
repair, and the title and composer written into the finished file. Each case
pins down a bug that came up while building the pipeline. It runs in about a
second and only needs numpy, so it can run before any transcription:

    python test_passes.py
"""
import sys, numpy as np

# Load only the parts of pipeline.py that don't need any audio libraries.
_src = open("pipeline.py").read()
NS = {"np": np, "os": __import__("os")}
exec(_src[_src.index("SR  = 16000"):_src.index("\ndef hz_to_midi(")], NS)
exec(_src[_src.index("MIN_DUR      = 0.08"):_src.index("\n", _src.index("MIN_DUR      = 0.08"))], NS)
for _fn in ("def merge_same_pitch(", "def merge_same_pitch_grid(", "def consolidate_melody(",
            "def clean_melody_rhythm(", "def declutter_zigzag_bars(", "def lift_low_treble_runs(",
            "def _patch_holes(", "def absorb_micro_rests(", "def even_out_snapped_pairs(",
            "def fit_uniform_grid(", "def repair_beat_grid(", "def set_score_info("):
    _i = _src.index(_fn)
    exec(_src[_i:_src.index("\ndef ", _i + 1)], NS)

merge_same_pitch      = NS["merge_same_pitch"]
consolidate_melody    = NS["consolidate_melody"]
merge_same_pitch_grid = NS["merge_same_pitch_grid"]
clean_melody_rhythm   = NS["clean_melody_rhythm"]
declutter_zigzag_bars = NS["declutter_zigzag_bars"]
lift_low_treble_runs  = NS["lift_low_treble_runs"]
_patch_holes          = NS["_patch_holes"]
fit_uniform_grid      = NS["fit_uniform_grid"]
repair_beat_grid      = NS["repair_beat_grid"]
set_score_info        = NS["set_score_info"]

FAILS = []

def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        expected {want}")
        print(f"        got      {got}")
        FAILS.append(name)

def ev(seq, start=0):
    """[(duration, pitch), ...] -> back-to-back events starting at `start`."""
    out, on = [], start
    for dur, p in seq:
        out.append({"on": on, "dur": dur, "kind": "note", "payload": p}); on += dur
    return out

def durs(events):  return [e["dur"] for e in events]
def pitches(events): return [e["payload"] for e in events]


print("\nSame-pitch merge")
# One held note that keeps re-triggering shows up as a stream of sixteenths at
# the same pitch, which looks like a burst of nonsense on the page.
check("stream of six sixteenths becomes eighths",
      durs(consolidate_melody(ev([(1, 71)] * 6))), [2, 2, 2])
check("stream of four sixteenths becomes eighths",
      durs(consolidate_melody(ev([(1, 71)] * 4))), [2, 2])
check("odd stream leaves its last fragment alone",
      durs(consolidate_melody(ev([(1, 71)] * 3))), [2, 1])
check("fragment merging never exceeds FRAGMENT_MERGE_MAX",
      max(durs(consolidate_melody(ev([(1, 71)] * 12)))), NS["FRAGMENT_MERGE_MAX"])

# A fragment chipped off a longer note gets absorbed back into it.
check("fragment after a long note is absorbed",
      durs(consolidate_melody(ev([(6, 71), (1, 71)]))), [7])
check("fragment before a long note is absorbed",
      durs(consolidate_melody(ev([(1, 71), (6, 71)]))), [7])

# A repeated syllable was sung twice, so both notes have to stay.
check("two full notes at one pitch both survive",
      durs(consolidate_melody(ev([(4, 71), (4, 71)]))), [4, 4])
check("three repeated eighths at one pitch all survive",
      durs(consolidate_melody(ev([(2, 71), (2, 71), (2, 71)]))), [2, 2, 2])

# Different pitches never merge, however short they are.
check("sixteenth run at different pitches untouched",
      durs(consolidate_melody(ev([(1, 71), (1, 73), (1, 74), (1, 76)]))), [1, 1, 1, 1])
check("pitches preserved through a merge",
      pitches(consolidate_melody(ev([(1, 71)] * 4))), [71, 71])

# A gap wider than MERGE_GAP_UNITS is a real rest between two notes.
gapped = [{"on": 0, "dur": 1, "kind": "note", "payload": 71},
          {"on": 4, "dur": 1, "kind": "note", "payload": 71}]
check("fragments separated by a rest stay separate",
      len(merge_same_pitch(gapped, 0)), 2)


print("\nMelody quantization")
check("genuine sixteenth run keeps its resolution",
      durs(clean_melody_rhythm(ev([(1, 71), (1, 73), (1, 74), (1, 76)]))), [1, 1, 1, 1])
check("longer note behind a run does not inherit the fine grid",
      durs(clean_melody_rhythm(ev([(1, 71), (1, 73), (4, 74)]))), [1, 1, 4])
check("isolated short note snaps to the eighth grid",
      durs(clean_melody_rhythm(ev([(1, 71), (3, 74)]))), [2, 2])


print("\nBar decluttering")
# CREPE jumping between two stacked harmony voices: dense, big leaps, and the
# direction flips on almost every note.
spray = ev([(1, 60), (1, 72), (1, 61), (1, 73), (1, 60), (1, 71), (1, 62), (1, 74)])
check("stacked-voice spray is thinned", len(declutter_zigzag_bars(spray)) < 8, True)
# A fast repeated figure is just as dense but has no leaps at all.
hook = ev([(1, 67)] * 8)
check("fast repeated figure is left alone",
      len(declutter_zigzag_bars(hook)), 8)
# A melisma run is dense but only moves in one direction.
run = ev([(1, 60 + k) for k in range(8)])
check("melisma run is left alone", len(declutter_zigzag_bars(run)), 8)


print("\nRegister repair")
low_run = ev([(2, 50), (2, 52), (2, 53), (2, 51)])
check("low run is lifted as one block",
      pitches(lift_low_treble_runs(low_run)), [62, 64, 65, 63])
check("isolated low note is not lifted",
      pitches(lift_low_treble_runs(ev([(2, 54), (2, 72)]))), [54, 72])
check("notes above the threshold never move",
      pitches(lift_low_treble_runs(ev([(2, 72), (2, 74)]))), [72, 74])


print("\nVocal gap handling")
# A gap in an instrumental passage belongs to stage 7. Patching it from the
# vocal stem writes bleed onto the page and also closes the gap, so stage 7
# never gets to fill it.
notes = [(20.0, 21.0, 60), (30.0, 31.0, 62)]
line  = [(1.0, 2.0, 80), (3.0, 4.0, 82), (24.0, 25.0, 64)]
holes, added, skipped = _patch_holes(notes, line, 2.0, 31.0, is_vocal=lambda a, b: a >= 10.0)
check("instrumental gap is left open", skipped, [(0.0, 20.0)])
check("nothing from an instrumental gap is written", [n for n in added if n[0] < 10.0], [])
check("vocal gap is still filled", added, [(24.0, 25.0, 64)])
holes, added, skipped = _patch_holes(notes, line, 2.0, 31.0, is_vocal=None)
check("without the test every gap is filled", len(added), 3)
check("added notes never overlap an existing note",
      all(e <= 20.0 or s >= 21.0 for s, e, _ in added), True)


print("\nBeat grid")
# madmom dropped beats in a quiet stretch, and every dropped beat made the
# score one beat shorter. A 128 BPM track with 15 beats missing in one spot:
rng = np.random.default_rng(0)
true_beats = np.arange(457) * (60.0 / 128) + rng.normal(0, 0.01, 457)
tracked = np.delete(true_beats, np.arange(300, 330, 2))
grid, used, coverage, bpm = fit_uniform_grid(tracked)
check("dropped beats in a quiet passage are put back", (used, len(grid)), (True, 457))
check("rebuilt grid lands on the right tempo", round(bpm), 128)
# A song that speeds up has no single steady pulse, so its beats are kept.
speeding_up = np.cumsum(60.0 / np.linspace(96, 104, 400))
check("a song that speeds up keeps its tracked beats",
      fit_uniform_grid(speeding_up)[1], False)
# The one-gap-at-a-time fallback still fills a single skipped beat.
fixed, inserted = repair_beat_grid(np.delete(np.arange(40) * 0.5, 20))
check("one skipped beat is filled back in", (inserted, len(fixed)), (1, 40))


print("\nTitle and composer")
# music21 leaves "Music21 Fragment" and "Music21" in the header, and MuseScore
# prints both at the top of the page.
import tempfile, os as _os
_head = ('<?xml version="1.0" encoding="utf-8"?>\n<score-partwise version="4.0">\n'
         "  <movement-title>Music21 Fragment</movement-title>\n  <identification>\n"
         '    <creator type="composer">Music21</creator>\n  </identification>\n'
         "  <part-list>keep me</part-list>\n</score-partwise>\n")
_p = _os.path.join(tempfile.mkdtemp(), "t.musicxml")
open(_p, "w").write(_head)
set_score_info(_p, "My Song", "A Name")
_out = open(_p).read()
check("title and composer are written in",
      ("<movement-title>My Song</movement-title>" in _out,
       '<creator type="composer">A Name</creator>' in _out), (True, True))
check("the music below the header is left alone", "<part-list>keep me</part-list>" in _out, True)
set_score_info(_p, "My Song", "")
check("a blank composer drops the line", "<creator" in open(_p).read(), False)
set_score_info(_p, "Tom & Jerry", "")
check("an ampersand in a title is escaped",
      "<movement-title>Tom &amp; Jerry</movement-title>" in open(_p).read(), True)


print()
if FAILS:
    print(f"{len(FAILS)} test(s) failed: " + ", ".join(FAILS))
    sys.exit(1)
print("All tests passed.")
