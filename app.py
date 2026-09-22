# app.py -- web demo for PyScribe (Gradio)
#
# All the real work happens in pipeline.run_pipeline(). This file takes the
# upload, runs the pipeline, and hands back the MusicXML file. Run it with
# `python app.py`, or put it on a Hugging Face Space next to pipeline.py.
#
# Uploads are saved as <name>_<hash8>.<ext>. The pipeline caches everything by
# song name, so two different files that are both called "song.mp3" would
# otherwise share stems. The hash comes from the file itself, so uploading the
# same file twice still reuses the cache.

import io
import os
import shutil
import hashlib
import traceback
import contextlib

import numpy as np
import gradio as gr

from pipeline import run_pipeline

INPUT_DIR     = os.path.join("data", "input")
OUTPUT_DIR    = "output"                            # pipeline writes output/<song>/
EXAMPLE_AUDIO = os.path.join("examples", "lift_me_up.mp3")
GITHUB_URL    = "https://github.com/Bomberman1359/PyScribe"
MAX_MINUTES   = 6   # the free CPU server is slow, so longer songs get turned away


def _file_hash(path, n=8):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


def _save_upload(uploaded_path):
    """Copy the upload into data/input/ as <name>_<hash8><ext>."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(uploaded_path))
    stem = "".join(c for c in stem if c not in '\\/:*?"<>|').strip() or "song"
    dest = os.path.join(INPUT_DIR, f"{stem}_{_file_hash(uploaded_path)}{ext or '.mp3'}")
    if not os.path.exists(dest):
        shutil.copyfile(uploaded_path, dest)
    return dest


def _check_length(path):
    """Turn away songs longer than MAX_MINUTES without loading the whole file."""
    import soundfile as sf
    try:
        info = sf.info(path)
        minutes = info.frames / info.samplerate / 60.0
    except Exception:
        return  # soundfile can't read this format; Demucs and librosa still can
    if minutes > MAX_MINUTES:
        raise gr.Error(
            f"That file is {minutes:.1f} minutes long. The demo only takes songs up to "
            f"{MAX_MINUTES} minutes so the free server doesn't time out. For longer "
            f"songs, run the pipeline on your own computer (see the GitHub repo)."
        )


def transcribe(audio_file, title, composer, seed_text, force, progress=gr.Progress()):
    if audio_file is None:
        raise gr.Error("Upload an audio file first (mp3 or wav).")

    # The saved file name carries a hash, so the title falls back to the name
    # of the file the visitor actually uploaded.
    title = (title or "").strip() or os.path.splitext(os.path.basename(audio_file))[0]

    # seed: blank = default sheet, "r" = random, anything else has to be a number
    seed = None
    s = (seed_text or "").strip().lower()
    if s == "r":
        seed = int(np.random.default_rng().integers(1, 100000))
    elif s:
        try:
            seed = int(s)
        except ValueError:
            raise gr.Error("The seed has to be a number, r for random, or blank.")

    _check_length(audio_file)
    audio_path = _save_upload(audio_file)

    progress(0.05, desc="Running the pipeline (stem separation takes the longest)...")

    # The pipeline prints a report for every stage; that report becomes the log.
    # Demucs runs as its own process, so its progress bar only shows in the
    # server console.
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            final_xml = run_pipeline(audio_path, force=bool(force), seed=seed,
                                     title=title, composer=composer)
    except Exception:
        return None, log.getvalue() + "\n\nERROR:\n" + traceback.format_exc()

    text = log.getvalue()
    if seed is not None:
        text += f"\n(seed used: {seed})\n"
    text += "\nDownload the file above and open it in MuseScore or any notation app."
    return final_xml, text


with gr.Blocks(title="PyScribe") as demo:
    gr.Markdown(
        f"""
        # PyScribe: audio to piano sheet music
        Upload a song and get back a two-hand piano arrangement as a MusicXML file,
        which opens in MuseScore (free) or any other notation app. The melody comes
        from the vocals, the chords come from the bass and backing instruments, and
        the left hand gets busier or calmer depending on the drums.

        **Heads up:** this runs on a small free server with no GPU, so a 3-minute
        song takes a while (the log shows each stage once it's done). It runs a lot
        faster on your own computer. The code is on [GitHub]({GITHUB_URL}).
        """
    )
    with gr.Row():
        with gr.Column():
            audio_in = gr.File(label="Song (mp3 or wav)", file_types=["audio"], type="filepath")
            title_in = gr.Textbox(label="Title", placeholder="printed at the top of the sheet (blank = the file name)")
            comp_in  = gr.Textbox(label="Composer", placeholder="printed under the title (blank = left off)")
            seed_in  = gr.Textbox(label="Seed (optional)",
                                  placeholder="blank = default sheet, a number = a variation you can repeat, r = random")
            force_in = gr.Checkbox(label="Rebuild everything from scratch (ignore cached results)", value=False)
            run_btn  = gr.Button("Transcribe", variant="primary")
            if os.path.isfile(EXAMPLE_AUDIO):
                gr.Examples(examples=[[EXAMPLE_AUDIO, "Lift Me Up", "Chunhwee Choi", "2", False]],
                            inputs=[audio_in, title_in, comp_in, seed_in, force_in],
                            label="Or try Lift Me Up, a song I wrote (seed 2, same as the example sheet on GitHub)",
                            cache_examples=False)
        with gr.Column():
            xml_out = gr.File(label="Sheet music (.musicxml)")
            log_out = gr.Textbox(label="Pipeline log", lines=18, max_lines=40)

    run_btn.click(transcribe, inputs=[audio_in, title_in, comp_in, seed_in, force_in],
                  outputs=[xml_out, log_out])

    gr.Markdown(
        "Only upload audio you have the rights to. Uploads sit in the server's "
        "temporary storage and get wiped whenever the Space restarts."
    )

# queue() runs one job at a time, which the pipeline needs since jobs share folders
demo.queue(max_size=8).launch(allowed_paths=[OUTPUT_DIR])
