# app.py -- web demo for the piano transcription pipeline (Gradio).
#
# This is a thin wrapper: all real work happens in pipeline.run_pipeline(),
# which is imported unchanged. Runs locally with `python app.py` and deploys
# as-is to a Hugging Face Space (Gradio SDK).
#
# It needs pipeline.py (and anything pipeline.py imports) in the same folder.

import io
import os
import shutil
import traceback
import contextlib

import numpy as np
import gradio as gr

from pipeline import run_pipeline   # <- your existing entry point, untouched

INPUT_DIR   = os.path.join("data", "input")
MAX_MINUTES = 6   # guard: free CPU hosting is slow; long files would time out


def _save_upload(uploaded_path: str) -> str:
    """Copy the uploaded temp file into data/input/ under its original name so
    the pipeline's <song> naming (and all caching) behaves exactly as it does
    locally. Returns the destination path."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    name = os.path.basename(uploaded_path)
    # strip characters that are awkward in shell/paths but keep spaces --
    # the pipeline already handles names like "Lift Me Up"
    safe = "".join(c for c in name if c not in '\\/:*?"<>|').strip() or "song.mp3"
    dest = os.path.join(INPUT_DIR, safe)
    shutil.copyfile(uploaded_path, dest)
    return dest


def _check_length(path: str):
    """Reject files longer than MAX_MINUTES without loading the whole file."""
    import soundfile as sf
    try:
        info = sf.info(path)
        minutes = info.frames / info.samplerate / 60.0
    except Exception:
        return  # not a format soundfile reads (e.g. mp3 on some builds); let
                # the pipeline handle it -- Demucs/librosa will load it fine
    if minutes > MAX_MINUTES:
        raise gr.Error(
            f"That file is {minutes:.1f} minutes long. The demo caps songs at "
            f"{MAX_MINUTES} minutes so the free server doesn't time out. "
            f"Run the pipeline locally (see the GitHub repo) for longer audio."
        )


def transcribe(audio_file, seed_text, force, progress=gr.Progress()):
    if audio_file is None:
        raise gr.Error("Upload an audio file first (mp3 or wav).")

    # --- parse seed: blank = deterministic default, "r" = random, else int ---
    seed = None
    s = (seed_text or "").strip().lower()
    if s == "r":
        seed = int(np.random.default_rng().integers(1, 100000))
    elif s:
        try:
            seed = int(s)
        except ValueError:
            raise gr.Error("Seed must be a number, 'r' for random, or blank.")

    audio_path = _save_upload(audio_file)
    _check_length(audio_path)

    progress(0.05, desc="Starting pipeline (stem separation is the slow part)...")

    # run_pipeline prints its stage-by-stage report; capture it as the log
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            final_xml = run_pipeline(audio_path, force=bool(force), seed=seed)
    except Exception:
        text = log.getvalue() + "\n\nERROR:\n" + traceback.format_exc()
        return None, text

    text = log.getvalue()
    if seed is not None:
        text += f"\n(seed used: {seed})\n"
    text += "\nDownload the file above and open it in MuseScore (or any MusicXML editor)."
    return final_xml, text


with gr.Blocks(title="Audio → Piano Sheet Music") as demo:
    gr.Markdown(
        """
        # Audio → Piano Sheet Music
        Upload a song and get back a playable grand-staff piano arrangement as
        **MusicXML** (opens in MuseScore, free). The pipeline separates the stems,
        tracks the vocal melody, recognizes the chords, and writes an accompaniment
        whose busyness follows the energy of the actual recording.

        **Heads up:** this runs real ML models (Demucs, CREPE) on a small server —
        a 3-minute song takes **on the order of 10–20 minutes** here. The same code
        runs much faster locally: see the [GitHub repo](https://github.com/YOUR_USERNAME/YOUR_REPO).
        """
    )
    with gr.Row():
        with gr.Column():
            audio_in  = gr.File(label="Song (mp3 / wav)", file_types=["audio"], type="filepath")
            seed_in   = gr.Textbox(label="Seed (optional)", placeholder="blank = default sheet · number = repeatable variant · r = random")
            force_in  = gr.Checkbox(label="Force full regeneration (ignore caches)", value=False)
            run_btn   = gr.Button("Transcribe", variant="primary")
        with gr.Column():
            xml_out   = gr.File(label="Sheet music (.musicxml)")
            log_out   = gr.Textbox(label="Pipeline log", lines=18, max_lines=40)

    run_btn.click(transcribe, inputs=[audio_in, seed_in, force_in], outputs=[xml_out, log_out])

    gr.Markdown(
        "Only upload audio you have the right to use. Nothing is stored after "
        "processing beyond the server's temporary working files."
    )

# queue() serializes jobs so two visitors can't overload the box
demo.queue(max_size=8).launch()
