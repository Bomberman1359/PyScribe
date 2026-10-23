# modal_app.py -- the PyScribe web app, hosted on Modal
#
# app.py is the version you run on your own computer. This file puts the same
# page online. The page runs in a small CPU container, and each song is sent to
# its own GPU container. Modal cuts off any web request after 150 seconds, and
# a song takes longer than that, so instead of waiting on one long request the
# page checks on the song every few seconds.
#
# Everything here shuts off when nobody is using it, and the daily song limit
# keeps one busy day from using up the month's free credits.
#
# Deploy from this folder:  modal deploy modal_app.py

import os
import json
import time
import hashlib

import modal

HERE        = os.path.dirname(os.path.abspath(__file__))
APP_DIR     = "/root/pyscribe"       # where the code sits inside the containers
GITHUB_URL  = "https://github.com/Bomberman1359/PyScribe"
MAX_MINUTES = 6       # longer songs get turned away
MAX_WAITING = 3       # songs allowed in line before new ones get turned away
MAX_PER_DAY = 12      # songs started per day (UTC); each one uses about 7 cents of credits
CHECK_EVERY = 5       # seconds between checks on a running song
GIVE_UP_MIN = 25      # the page stops waiting on a song after this many minutes

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libsndfile1", "git", "build-essential")
    .pip_install_from_requirements(os.path.join(HERE, "requirements.txt"))
    # Demucs downloads its model the first time it runs. Doing that here puts
    # the model in the image, so no song has to wait for the download.
    .run_commands("python -c \"from demucs.pretrained import get_model; get_model('htdemucs')\"")
    .add_local_file(os.path.join(HERE, "pipeline.py"), f"{APP_DIR}/pipeline.py")
    .add_local_file(os.path.join(HERE, "examples", "lift_me_up.mp3"),
                    f"{APP_DIR}/examples/lift_me_up.mp3")
)

app = modal.App("pyscribe", image=image)

# Small shared notes between the containers: the stage each running song is on
# ("[3-4/11] Vocal transcription") and how many songs were started today.
progress = modal.Dict.from_name("pyscribe-progress", create_if_missing=True)


# ============================================================================
# GPU side: one song, start to finish
# ============================================================================
class _StageLog:
    """Takes the place of stdout while the pipeline runs. Keeps the whole
    printout for the log box and hands each stage header to `report`."""

    def __init__(self, report):
        self.report, self.parts, self.line = report, [], ""

    def write(self, text):
        self.parts.append(text)
        self.line += text
        while "\n" in self.line:
            done, self.line = self.line.split("\n", 1)
            if done.startswith("[") and "/11]" in done:
                self.report(done)
        return len(text)

    def flush(self):
        pass

    def getvalue(self):
        return "".join(self.parts)


def run_job(filename, audio, seed, title, composer, workdir, report=lambda line: None):
    """Save the upload as workdir/data/input/<filename> and run the pipeline.
    Returns {"xml": bytes or None, "name": file name, "log": the printout}."""
    import sys
    import contextlib
    import traceback

    os.chdir(workdir)
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    path = os.path.join("data", "input", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):          # the name has the file's hash in it
        with open(path, "wb") as f:
            f.write(audio)

    log = _StageLog(report)
    started = time.time()
    try:
        from pipeline import run_pipeline
        with contextlib.redirect_stdout(log):
            final_xml = run_pipeline(path, seed=seed, title=title, composer=composer)
        with open(final_xml, "rb") as f:
            return {"xml": f.read(), "name": os.path.basename(final_xml),
                    "log": log.getvalue(), "seconds": time.time() - started}
    except Exception:
        return {"xml": None, "name": None,
                "log": log.getvalue() + "\n\nERROR:\n" + traceback.format_exc()}


def _report(job_id, line):
    try:
        progress[job_id] = line
    except Exception:
        pass    # the stage display is extra; the song keeps going without it


# One GPU container at most, no retries, a hard stop after 20 minutes, and it
# shuts off 30 seconds after the last song.
@app.function(gpu="T4", cpu=2.0, memory=8192, timeout=20 * 60,
              max_containers=1, scaledown_window=30)
def transcribe_song(filename: str, audio: bytes, seed, title: str, composer: str,
                    job_id: str):
    return run_job(filename, audio, seed, title, composer, APP_DIR,
                   report=lambda line: _report(job_id, line))


# ============================================================================
# Web side: the page
# ============================================================================
def _today():
    return "songs " + time.strftime("%Y-%m-%d", time.gmtime())


class ModalJobs:
    """Hands songs to transcribe_song and checks on them."""

    def full_today(self):
        try:
            return progress.get(_today(), 0) >= MAX_PER_DAY
        except Exception:
            return False

    def busy(self):
        try:
            return transcribe_song.get_current_stats().backlog >= MAX_WAITING
        except Exception:
            return False

    def start(self, filename, audio, seed, title, composer, job_id):
        call = transcribe_song.spawn(filename, audio, seed, title, composer, job_id)
        try:
            progress[_today()] = progress.get(_today(), 0) + 1
        except Exception:
            pass
        return call.object_id

    def check(self, call_id, job_id):
        """("running", stage or None) while the song is going, ("done", result) after."""
        try:
            return "done", modal.FunctionCall.from_id(call_id).get(timeout=0)
        except modal.exception.OutputExpiredError:
            return "done", {"xml": None, "name": None,
                            "log": "The result expired before the page picked it up. Run it again."}
        except TimeoutError:
            try:
                return "running", progress.get(job_id)
            except Exception:
                return "running", None
        except Exception as ex:
            return "done", {"xml": None, "name": None,
                            "log": f"The GPU server stopped with an error: {ex!r}"}

    def forget(self, job_id):
        try:
            progress.pop(job_id)
        except Exception:
            pass


def _parse_seed(seed_text):
    """blank = default sheet, "r" = random, anything else has to be a number"""
    import gradio as gr
    import numpy as np
    s = (seed_text or "").strip().lower()
    if not s:
        return None
    if s == "r":
        return int(np.random.default_rng().integers(1, 100000))
    try:
        return int(s)
    except ValueError:
        raise gr.Error("The seed has to be a number, r for random, or blank.")


def _check_length(path):
    """Turn away songs longer than MAX_MINUTES without loading the whole file."""
    import gradio as gr
    import soundfile as sf
    try:
        info = sf.info(path)
        minutes = info.frames / info.samplerate / 60.0
    except Exception:
        return  # soundfile can't read this format; Demucs still can
    if minutes > MAX_MINUTES:
        raise gr.Error(
            f"That file is {minutes:.1f} minutes long. This site takes songs up to "
            f"{MAX_MINUTES} minutes. For longer songs, run PyScribe on your own "
            f"computer (see the GitHub repo)."
        )


def build_ui(jobs):
    import gradio as gr

    def start(audio_file, title, composer, seed_text):
        if audio_file is None:
            raise gr.Error("Upload an audio file first (mp3 or wav).")
        seed = _parse_seed(seed_text)
        _check_length(audio_file)
        if jobs.full_today():
            raise gr.Error(f"This site has hit its limit of {MAX_PER_DAY} songs for today. It runs on "
                           "free credits, so come back tomorrow or run PyScribe on your own computer.")
        if jobs.busy():
            raise gr.Error("A few songs are already waiting. Try again in a few minutes.")
        try:
            with open(audio_file, "rb") as f:
                audio = f.read()
        except OSError:      # the page's server restarted since the upload
            raise gr.Error("That upload is no longer on the server. Upload the file again.")
        # Same naming as app.py: the file's hash keeps two different "song.mp3"
        # uploads apart, since the pipeline caches everything by song name.
        stem, ext = os.path.splitext(os.path.basename(audio_file))
        stem = "".join(c for c in stem if c not in '\\/:*?"<>|').strip() or "song"
        filename = f"{stem}_{hashlib.sha1(audio).hexdigest()[:8]}{ext or '.mp3'}"
        title = (title or "").strip() or stem
        job_id = hashlib.sha1(f"{filename} {time.time()}".encode()).hexdigest()[:12]
        try:
            call_id = jobs.start(filename, audio, seed, title, (composer or "").strip(), job_id)
        except Exception:
            raise gr.Error("The GPU server didn't take the song. This site may be out of "
                           "free credits for this month.")
        job = json.dumps({"call": call_id, "id": job_id, "start": time.time(), "seed": seed})
        return (job, gr.Timer(active=True), gr.Button(interactive=False), None,
                "Sent to the GPU server.")

    def check(job_text):
        if not job_text:
            return "", gr.Timer(active=False), gr.Button(interactive=True), gr.skip(), gr.skip()
        job = json.loads(job_text)
        state, info = jobs.check(job["call"], job["id"])
        waited = int(time.time() - job["start"])
        if state == "running":
            if waited > GIVE_UP_MIN * 60:
                jobs.forget(job["id"])
                return ("", gr.Timer(active=False), gr.Button(interactive=True), None,
                        "This song took too long, so the page stopped waiting. Try again later.")
            stage = info or "waiting for the GPU server to start"
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    f"Working on it ({waited // 60}:{waited % 60:02d} so far)\nNow on: {stage}")

        jobs.forget(job["id"])
        text = info.get("log") or ""
        if job.get("seed") is not None:
            text += f"\n(seed used: {job['seed']})\n"
        if not info.get("xml"):
            return "", gr.Timer(active=False), gr.Button(interactive=True), None, text
        folder = os.path.join("output", job["id"])
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, info["name"])
        with open(path, "wb") as f:
            f.write(info["xml"])
        took = int(info.get("seconds") or waited)
        text += (f"\nDone in {took // 60}:{took % 60:02d} on the GPU server. Download the file above "
                 f"and open it in MuseScore or any notation app.")
        return "", gr.Timer(active=False), gr.Button(interactive=True), path, text

    example = os.path.join("examples", "lift_me_up.mp3")
    with gr.Blocks(title="PyScribe") as demo:
        gr.Markdown(
            f"""
            # PyScribe: audio to piano sheet music
            Upload a song and get back a two-hand piano arrangement as a MusicXML file,
            which opens in MuseScore (free) or any other notation app. The melody comes
            from the vocals, the chords come from the bass and backing instruments, and
            the left hand gets busier or calmer depending on the drums.

            **Heads up:** a 3-minute song takes about 3 to 4 minutes on the GPU server.
            Keep this tab open until it's done. The log shows which stage it's on.
            Songs can be up to {MAX_MINUTES} minutes long, and the site takes
            {MAX_PER_DAY} songs a day because it runs on free credits.
            The code is on [GitHub]({GITHUB_URL}).
            """
        )
        # The running song's ticket is kept in the page itself (a hidden box), not
        # on the server. That way the page's server holds nothing between clicks
        # and can shut off as soon as nobody is using it.
        job = gr.Textbox(visible=False)
        timer = gr.Timer(CHECK_EVERY, active=False)
        with gr.Row():
            with gr.Column():
                audio_in = gr.File(label="Song (mp3 or wav)", file_types=["audio"], type="filepath")
                title_in = gr.Textbox(label="Title", placeholder="printed at the top of the sheet (blank = the file name)")
                comp_in  = gr.Textbox(label="Composer", placeholder="printed under the title (blank = left off)")
                seed_in  = gr.Textbox(label="Seed (optional)",
                                      placeholder="blank = default sheet, a number = a variation you can repeat, r = random")
                run_btn  = gr.Button("Transcribe", variant="primary")
                if os.path.isfile(example):
                    gr.Examples(examples=[[example, "Lift Me Up", "Chunhwee Choi", "2"]],
                                inputs=[audio_in, title_in, comp_in, seed_in],
                                label="Or try Lift Me Up, a song I wrote (seed 2, the same settings as the example sheet on GitHub)",
                                cache_examples=False)
            with gr.Column():
                xml_out = gr.File(label="Sheet music (.musicxml)")
                log_out = gr.Textbox(label="Pipeline log", lines=18, max_lines=40)

        run_btn.click(start, inputs=[audio_in, title_in, comp_in, seed_in],
                      outputs=[job, timer, run_btn, xml_out, log_out])
        timer.tick(check, inputs=[job], outputs=[job, timer, run_btn, xml_out, log_out])

        gr.Markdown(
            "Only upload audio you have the rights to. Files only live on temporary "
            "servers that get wiped when they shut down."
        )
    return demo


def make_web_app(jobs, workdir):
    import gradio as gr
    from fastapi import FastAPI
    os.chdir(workdir)
    demo = build_ui(jobs).queue(max_size=16, default_concurrency_limit=4)
    return gr.mount_gradio_app(FastAPI(), demo, path="/", max_file_size="80mb",
                               allowed_paths=[os.path.join(workdir, "output")])


# Gradio needs every visitor on the same container, hence max_containers=1.
# It's a quarter of a CPU core, and it shuts off 5 minutes after the last click.
@app.function(cpu=0.25, memory=512, max_containers=1, scaledown_window=300)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    return make_web_app(ModalJobs(), APP_DIR)
