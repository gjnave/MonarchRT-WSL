"""
Self-Forcing Video Generation — Gradio Interface
Uses inference_server.py as a persistent model process.
Model loads ONCE; subsequent generations take ~14s instead of minutes.
"""

import gradio as gr
import subprocess
import os
import glob
import uuid
import json
import threading
import time
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG / CHECKPOINT DISCOVERY
# ─────────────────────────────────────────────

def discover_configs():
    configs = glob.glob("configs/*.yaml")
    configs = sorted(configs) if configs else ["configs/self_forcing_monarch_dmd.yaml"]
    preferred = [c for c in configs if "monarch" in c.lower()]
    return preferred + [c for c in configs if c not in preferred]

def discover_checkpoints():
    ckpts = glob.glob("checkpoints/*.pt") + glob.glob("checkpoints/*.safetensors")
    ckpts = sorted(ckpts) if ckpts else ["checkpoints/self_forcing_monarch_dmd.pt"]
    preferred = [c for c in ckpts if "monarch" in c.lower()]
    return preferred + [c for c in ckpts if c not in preferred]

# ─────────────────────────────────────────────
# PERSISTENT SERVER MANAGER
# ─────────────────────────────────────────────

class InferenceServer:
    """
    Manages a single long-lived inference_server.py subprocess.
    The model is loaded once; jobs are sent via stdin as JSON lines.
    Responses come back via stdout as JSON lines.
    """

    def __init__(self):
        self.process    = None
        self.lock       = threading.Lock()
        self.config     = None   # track what config/ckpt the server was launched with
        self.checkpoint = None
        self.use_ema    = None

    def _is_alive(self):
        return self.process is not None and self.process.poll() is None

    def start(self, config_path, checkpoint_path, use_ema,
              use_torch_compile, disable_offload, seed, log_cb):
        """Launch the server process. Blocks until it emits {"type":"ready"}."""

        if self._is_alive():
            # Already running with same settings — reuse it
            if (self.config == config_path and
                    self.checkpoint == checkpoint_path and
                    self.use_ema == use_ema):
                log_cb("SERVER: model already loaded, skipping reload.")
                return True
            else:
                log_cb("SERVER: config changed, restarting server...")
                self.stop()

        cmd = [
            "python", "-u", "inference_server.py",
            "--config_path", config_path,
            "--seed", str(seed),
        ]
        if checkpoint_path:
            cmd += ["--checkpoint_path", checkpoint_path]
        if use_ema:
            cmd.append("--use_ema")
        if use_torch_compile:
            cmd.append("--use_torch_compile")
        if disable_offload:
            cmd.append("--disable_offload")

        log_cb(f"SERVER: launching — {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        # Wait for "ready" signal, streaming logs meanwhile
        for raw in iter(self.process.stdout.readline, ""):
            line = raw.rstrip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if msg.get("type") == "ready":
                    log_cb("SERVER: model ready.")
                    self.config     = config_path
                    self.checkpoint = checkpoint_path
                    self.use_ema    = use_ema
                    return True
                elif msg.get("type") == "log":
                    log_cb(f"[init] {msg['msg']}")
                elif msg.get("type") == "error":
                    log_cb(f"[init ERROR] {msg['msg']}")
                    return False
            except json.JSONDecodeError:
                log_cb(line)  # raw line (e.g. Python traceback)

            if self.process.poll() is not None:
                log_cb("SERVER: process exited during startup.")
                return False

        return False

    def generate(self, job: dict, log_cb):
        """
        Send a job to the running server and stream responses.
        Yields log strings. Returns list of video paths when done.
        """
        if not self._is_alive():
            raise RuntimeError("Server is not running. Start it first.")

        payload = json.dumps(job) + "\n"
        self.process.stdin.write(payload)
        self.process.stdin.flush()

        videos = []
        for raw in iter(self.process.stdout.readline, ""):
            line = raw.rstrip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                t = msg.get("type")
                if t == "log":
                    log_cb(msg["msg"])
                elif t == "done":
                    videos = msg["videos"]
                    break
                elif t == "error":
                    raise RuntimeError(msg["msg"])
            except json.JSONDecodeError:
                log_cb(line)

            if self.process.poll() is not None:
                raise RuntimeError("Server process died unexpectedly.")

        return videos

    def stop(self):
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process    = None
            self.config     = None
            self.checkpoint = None
            self.use_ema    = None

    @property
    def is_ready(self):
        return self._is_alive()


# Global server instance — persists across Gradio requests
SERVER = InferenceServer()

# ─────────────────────────────────────────────
# STATUS HTML HELPER
# ─────────────────────────────────────────────

_STAGES = [
    "Loading config",
    "Building pipeline",
    "Loading checkpoint",
    "Moving to device",
    "Model ready",
    "Encoding text",
    "Denoising",
    "Decoding video",
    "Saving",
]

def _detect_stage(log_lines):
    text = " ".join(log_lines).lower()
    stage = 0
    hints = [
        ("loading config",     0),
        ("building pipeline",  1),
        ("loading checkpoint", 2),
        ("moving model",       3),
        ("model ready",        4),
        ("model already",      4),
        ("generating",         5),
        ("encoding",           5),
        ("denoising",          6),
        ("running denois",     6),
        ("decoding",           7),
        ("saving",             8),
        ("saved:",             8),
    ]
    for kw, idx in hints:
        if kw in text:
            stage = max(stage, idx)
    return stage

def _status_html(state, elapsed, log_lines):
    stage_idx  = _detect_stage(log_lines)
    total      = len(_STAGES) - 1
    pct        = int((stage_idx / total) * 100) if state != "done" else 100
    stage_lbl  = _STAGES[min(stage_idx, total)]

    if state == "done":
        bar_col, dot_col, status_txt = "#47ffb3", "#47ffb3", "DONE"
        stage_lbl = "Complete"
    elif state == "running":
        bar_col, dot_col, status_txt = "#e8ff47", "#e8ff47", "RUNNING"
    elif state == "ready":
        bar_col, dot_col, status_txt = "#47b3ff", "#47b3ff", "MODEL LOADED"
        pct = 100
        stage_lbl = "Ready for generation"
    else:
        bar_col, dot_col, status_txt = "#2a2a35", "#7070a0", "IDLE"
        pct = 0

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    pips = ""
    for i in range(len(_STAGES)):
        if i < stage_idx:
            col = "#47ffb3"
        elif i == stage_idx:
            col = dot_col
        else:
            col = "#1e1e28"
        pips += f'''<div title="{_STAGES[i]}" style="width:7px;height:7px;border-radius:50%;background:{col};flex-shrink:0;transition:background 0.4s;"></div>'''

    return f'''
    <div style="background:#111116;border:1px solid #2a2a35;border-radius:6px;
                padding:14px 18px;font-family:\'DM Mono\',monospace;margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:7px;height:7px;border-radius:50%;background:{dot_col};
                            box-shadow:0 0 6px {dot_col};"></div>
                <span style="font-size:10px;letter-spacing:0.18em;color:{dot_col};">{status_txt}</span>
            </div>
            <span style="font-size:11px;color:#7070a0;">{elapsed_str}</span>
        </div>
        <div style="background:#0a0a0c;border-radius:3px;height:4px;width:100%;
                    margin-bottom:10px;overflow:hidden;">
            <div style="height:100%;width:{pct}%;background:{bar_col};border-radius:3px;
                        transition:width 0.5s ease;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="display:flex;gap:5px;align-items:center;">{pips}</div>
            <span style="font-size:10px;color:#e8e8f0;letter-spacing:0.06em;">{stage_lbl}</span>
        </div>
    </div>'''

# ─────────────────────────────────────────────
# INFERENCE HANDLER (generator — streams to UI)
# ─────────────────────────────────────────────

def run_inference(
    prompt, extended_prompt,
    config_path, checkpoint_path,
    num_samples, seed,
    use_ema, save_with_index,
    use_torch_compile, disable_offload,
):
    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty.")

    run_id        = str(uuid.uuid4())[:8]
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir    = f"outputs/run_{timestamp}_{run_id}"
    log_lines     = []
    start_time    = time.time()

    def log_cb(msg):
        log_lines.append(msg)

    def status():
        elapsed = time.time() - start_time
        state   = "done" if any("saved:" in l.lower() for l in log_lines) else "running"
        return _status_html(state, elapsed, log_lines)

    def partial(video=None, gallery=None):
        elapsed = time.time() - start_time
        return (
            video,
            gallery or [],
            "\n".join(log_lines[-120:]),
            output_dir,
            _status_html("running", elapsed, log_lines),
        )

    # ── Phase 1: ensure server is loaded ──────
    yield partial()

    with SERVER.lock:
        ok = SERVER.start(
            config_path, checkpoint_path,
            use_ema, use_torch_compile, disable_offload,
            seed, log_cb,
        )

    if not ok:
        raise gr.Error("Failed to start inference server. Check the log for details.")

    elapsed = time.time() - start_time
    yield (
        None, [],
        "\n".join(log_lines[-120:]),
        output_dir,
        _status_html("ready", elapsed, log_lines),
    )

    # ── Phase 2: generate ─────────────────────
    job = {
        "job_id":          run_id,
        "prompt":          prompt.strip(),
        "extended_prompt": extended_prompt.strip(),
        "output_dir":      output_dir,
        "num_samples":     num_samples,
        "seed":            seed,
    }

    result_videos = []

    def _generate():
        nonlocal result_videos
        with SERVER.lock:
            result_videos = SERVER.generate(job, log_cb)

    gen_thread = threading.Thread(target=_generate, daemon=True)
    gen_thread.start()

    # Stream log updates to UI while generation runs
    while gen_thread.is_alive():
        time.sleep(0.3)
        yield partial()

    gen_thread.join()

    if not result_videos:
        raise gr.Error("Generation completed but no videos were returned.")

    elapsed = time.time() - start_time
    yield (
        result_videos[0],
        result_videos,
        "\n".join(log_lines),
        output_dir,
        _status_html("done", elapsed, log_lines),
    )


# ─────────────────────────────────────────────
# SHUTDOWN HOOK
# ─────────────────────────────────────────────

import atexit
atexit.register(SERVER.stop)

# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0a0c;
    --bg-card:   #111116;
    --bg-input:  #18181f;
    --border:    #2a2a35;
    --accent:    #e8ff47;
    --accent-dim:#b8cc30;
    --text:      #e8e8f0;
    --text-muted:#7070a0;
    --danger:    #ff4d6d;
    --success:   #47ffb3;
    --mono:      'DM Mono', monospace;
    --sans:      'DM Sans', sans-serif;
    --display:   'Bebas Neue', sans-serif;
    --radius:    6px;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* ── Header ── */
.sf-header {
    padding: 48px 0 32px 0;
    text-align: left;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
.sf-header .eyebrow {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 8px;
}
.sf-header h1 {
    font-family: var(--display);
    font-size: clamp(52px, 8vw, 96px);
    letter-spacing: 0.04em;
    color: var(--text);
    line-height: 0.95;
    margin: 0 0 12px 0;
}
.sf-header h1 span {
    color: var(--accent);
}
.sf-header .subtitle {
    font-size: 14px;
    color: var(--text-muted);
    font-weight: 300;
}

/* ── Panels ── */
.panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
}
.panel-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-label::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
}

/* ── Inputs ── */
textarea, input[type="text"], input[type="number"] {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 14px !important;
    transition: border-color 0.2s !important;
}
textarea:focus, input:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(232,255,71,0.08) !important;
}
label span {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* ── Select / Dropdown ── */
select, .gradio-dropdown select {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* ── Sliders ── */
input[type="range"] {
    accent-color: var(--accent) !important;
}

/* ── Checkboxes ── */
input[type="checkbox"] {
    accent-color: var(--accent) !important;
}

/* ── Generate Button ── */
#generate-btn {
    background: var(--accent) !important;
    color: #0a0a0c !important;
    border: none !important;
    font-family: var(--display) !important;
    font-size: 22px !important;
    letter-spacing: 0.1em !important;
    padding: 16px 40px !important;
    border-radius: var(--radius) !important;
    cursor: pointer !important;
    transition: background 0.15s, transform 0.1s !important;
    width: 100% !important;
}
#generate-btn:hover {
    background: var(--accent-dim) !important;
    transform: translateY(-1px) !important;
}
#generate-btn:active {
    transform: translateY(0) !important;
}

/* ── Log box ── */
#log-box textarea {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: #90e0a0 !important;
    background: #080c08 !important;
    border: 1px solid #1a2a1a !important;
    line-height: 1.6 !important;
}

/* ── Video output ── */
video {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    width: 100% !important;
}

/* ── Gallery ── */
.gradio-gallery {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Output path ── */
#output-path input {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: var(--accent) !important;
}

/* ── Accordion ── */
.gradio-accordion {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-card) !important;
}
.gradio-accordion > .label-wrap {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    padding: 14px 18px !important;
}

/* ── Tabs ── */
.gradio-tabs .tab-nav button {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.gradio-tabs .tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Status badge ── */
.status-idle   { color: var(--text-muted); }
.status-run    { color: var(--accent); }
.status-done   { color: var(--success); }
.status-error  { color: var(--danger); }

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}


/* ── Progress status block ── */
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
.status-running-dot {
    animation: pulse-dot 1.2s ease-in-out infinite;
}

/* ── Footer ── */
.sf-footer {
    text-align: center;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-muted);
    padding: 32px 0 16px 0;
    letter-spacing: 0.1em;
}
"""

with gr.Blocks(title="Self-Forcing — Video Generation") as demo:

    gr.HTML("""
    <div class="sf-header">
        <div class="eyebrow">Wan2.1-T2V-14B · Monarch DMD · Persistent Server</div>
        <h1>SELF<span>FORCING</span></h1>
        <div class="subtitle">
            Model loads once &nbsp;·&nbsp; ~14s per generation &nbsp;·&nbsp;
            480×832px &nbsp;·&nbsp; 21 frames &nbsp;·&nbsp; 16fps
        </div>
        <div class="eyebrow"><a href="https://getgoingfast.pro">GET GOING FAST </a></div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT: controls ───────────────────────
        with gr.Column(scale=5):

            gr.HTML('<div class="panel-label">Prompt</div>')
            prompt = gr.Textbox(
                label="Text Prompt",
                placeholder="A cinematic drone shot over a misty mountain range at sunrise...",
                lines=3,
            )
            extended_prompt = gr.Textbox(
                label="Extended Prompt  (optional)",
                placeholder="Longer, richer description fed directly to the model. Short prompt used for file naming only.",
                lines=2,
            )

            gr.HTML('<hr class="divider">')
            gr.HTML('<div class="panel-label">Model</div>')

            with gr.Row():
                config_path = gr.Dropdown(
                    label="Config",
                    choices=discover_configs(),
                    value=discover_configs()[0],
                    allow_custom_value=True,
                )
                checkpoint_path = gr.Dropdown(
                    label="Checkpoint",
                    choices=discover_checkpoints(),
                    value=discover_checkpoints()[0],
                    allow_custom_value=True,
                )

            gr.HTML("""
            <div style="background:#18181f;border:1px solid #2a2a35;border-left:3px solid #e8ff47;
                        border-radius:6px;padding:12px 16px;font-family:\'DM Mono\',monospace;
                        font-size:11px;color:#7070a0;letter-spacing:0.08em;margin:8px 0 16px 0;">
                <span style="color:#e8ff47;">NOTE</span>&nbsp;
                Changing config or checkpoint will reload the model (~minutes).
                Same config = instant generation (~14s).
            </div>
            """)

            gr.HTML('<hr class="divider">')
            gr.HTML('<div class="panel-label">Generation</div>')

            gr.HTML("""
            <div style="background:#18181f;border:1px solid #2a2a35;border-left:3px solid #e8ff47;
                        border-radius:6px;padding:12px 16px;font-family:\'DM Mono\',monospace;
                        font-size:11px;color:#7070a0;letter-spacing:0.08em;margin-bottom:16px;">
                <span style="color:#e8ff47;">MODEL OUTPUT (fixed by config)</span><br>
                &nbsp;&nbsp;frames &nbsp;&nbsp;&nbsp;→&nbsp; 21 &nbsp;&nbsp;<span style="color:#e8e8f0;">(~1.3s @ 16fps)</span><br>
                &nbsp;&nbsp;resolution →&nbsp; 480×832px &nbsp;<span style="color:#e8e8f0;">(latent 60×104)</span><br>
                &nbsp;&nbsp;denoising &nbsp;→&nbsp; 4 steps &nbsp;<span style="color:#e8e8f0;">[1000→750→500→250]</span><br>
                &nbsp;&nbsp;guidance &nbsp;&nbsp;→&nbsp; 3.0 (cfg scale)
            </div>
            """)

            with gr.Row():
                num_samples = gr.Slider(
                    label="Samples per Prompt", minimum=1, maximum=8, step=1, value=1,
                )
            with gr.Row():
                seed = gr.Number(label="Seed", value=0, precision=0)

            with gr.Accordion("Advanced Options", open=False):
                with gr.Row():
                    use_ema = gr.Checkbox(label="Use EMA Weights", value=True,
                                          info="Recommended — better quality")
                    use_torch_compile = gr.Checkbox(label="torch.compile",
                                                     info="~5 min warmup, faster after")
                    disable_offload   = gr.Checkbox(label="Disable CPU Offload",
                                                     info="Needs 40+ GB VRAM")
                with gr.Row():
                    save_with_index = gr.Checkbox(label="Save by Index", value=True)

            gr.HTML('<hr class="divider">')
            generate_btn = gr.Button("GENERATE VIDEO", elem_id="generate-btn", variant="primary")

        # ── RIGHT: output ─────────────────────────
        with gr.Column(scale=7):

            gr.HTML('<div class="panel-label">Status</div>')
            status_block = gr.HTML(value='<div style="background:#111116;border:1px solid #2a2a35;border-radius:6px;padding:14px 18px;font-family:DM Mono,monospace;color:#7070a0;font-size:11px;letter-spacing:0.12em;">IDLE — waiting for generation</div>')

            gr.HTML('<div class="panel-label">Output</div>')
            with gr.Tabs():
                with gr.Tab("Preview"):
                    video_out = gr.Video(label="Generated Video", interactive=False, autoplay=True)
                with gr.Tab("All Samples"):
                    gallery_out = gr.Gallery(label="All Samples", columns=2,
                                             object_fit="contain", height=400)

            output_path_box = gr.Textbox(label="Output Folder", interactive=False,
                                         elem_id="output-path")

            gr.HTML('<div class="panel-label" style="margin-top:20px;">Console Log</div>')
            log_box = gr.Textbox(label="", lines=14, max_lines=14,
                                 interactive=False, elem_id="log-box")

    gr.HTML("""
    <div class="sf-footer">
        SELF-FORCING &nbsp;·&nbsp; WAN2.1-T2V-14B MONARCH DMD
        &nbsp;·&nbsp; PERSISTENT SERVER &nbsp;·&nbsp; WSL/LINUX
    </div>
    """)

    generate_btn.click(
        fn=run_inference,
        inputs=[
            prompt, extended_prompt,
            config_path, checkpoint_path,
            num_samples, seed,
            use_ema, save_with_index,
            use_torch_compile, disable_offload,
        ],
        outputs=[video_out, gallery_out, log_box, output_path_box, status_block],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CSS,
    )
