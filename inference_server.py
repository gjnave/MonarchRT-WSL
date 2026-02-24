"""
inference_server.py — Persistent model server for Self-Forcing video generation.

Loads the model ONCE, then reads newline-delimited JSON jobs from stdin
and writes newline-delimited JSON status messages to stdout.

Protocol:
  STDIN  (app → server):  {"job_id": "...", "prompt": "...", "extended_prompt": "...",
                            "output_path": "...", "num_samples": 1, "seed": 0}
  STDOUT (server → app):  {"type": "ready"}
                          {"type": "log",    "job_id": "...", "msg": "..."}
                          {"type": "done",   "job_id": "...", "videos": ["path1", ...]}
                          {"type": "error",  "job_id": "...", "msg": "..."}

The server runs forever until stdin is closed (i.e. the parent process exits).
"""

import argparse
import json
import os
import sys
import torch
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
    BidirectionalDiffusionInferencePipeline,
    BidirectionalInferencePipeline,
)
from utils.misc import set_seed
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

# Force stdout to be unbuffered so app.py sees lines immediately
sys.stdout.reconfigure(line_buffering=True)


def emit(obj: dict):
    """Write a JSON message to stdout, flushed immediately."""
    print(json.dumps(obj), flush=True)


def log(job_id: str, msg: str):
    emit({"type": "log", "job_id": job_id, "msg": msg})


# ─────────────────────────────────────────────
# ARG PARSE
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--config_path",     type=str, required=True)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--use_ema",         action="store_true")
parser.add_argument("--use_torch_compile", action="store_true")
parser.add_argument("--disable_offload", action="store_true")
parser.add_argument("--seed",            type=int, default=0)
args = parser.parse_args()

# ─────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────

device = torch.device("cuda")
set_seed(args.seed)

emit({"type": "log", "job_id": "__init__", "msg": f"Free VRAM: {get_cuda_free_memory_gb(gpu):.2f} GB"})
low_memory = (get_cuda_free_memory_gb(gpu) < 40) and (not args.disable_offload)
if low_memory:
    emit({"type": "log", "job_id": "__init__", "msg": "Low VRAM detected — CPU offloading enabled"})

torch.set_grad_enabled(False)

# ─────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────

emit({"type": "log", "job_id": "__init__", "msg": f"Loading config: {args.config_path}"})
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# ─────────────────────────────────────────────
# BUILD PIPELINE
# ─────────────────────────────────────────────

emit({"type": "log", "job_id": "__init__", "msg": "Building pipeline..."})
if hasattr(config, 'denoising_step_list'):
    pipeline = (CausalInferencePipeline(config, device=device)
                if config.causal else
                BidirectionalInferencePipeline(config, device=device))
else:
    pipeline = (CausalDiffusionInferencePipeline(config, device=device)
                if config.causal else
                BidirectionalDiffusionInferencePipeline(config, device=device))

# ─────────────────────────────────────────────
# LOAD CHECKPOINT
# ─────────────────────────────────────────────

if args.checkpoint_path:
    emit({"type": "log", "job_id": "__init__", "msg": f"Loading checkpoint: {args.checkpoint_path}"})
    if args.checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(args.checkpoint_path, device="cpu")
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    else:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        key = 'generator_ema' if args.use_ema else 'generator'
        state_dict = state_dict[key]
    pipeline.generator.load_state_dict(state_dict)
    emit({"type": "log", "job_id": "__init__", "msg": f"Checkpoint loaded ({'EMA' if args.use_ema else 'base'} weights)"})

# ─────────────────────────────────────────────
# MOVE TO DEVICE
# ─────────────────────────────────────────────

emit({"type": "log", "job_id": "__init__", "msg": "Moving model to device..."})
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)

# ─────────────────────────────────────────────
# OPTIONAL TORCH COMPILE
# ─────────────────────────────────────────────

if args.use_torch_compile:
    emit({"type": "log", "job_id": "__init__", "msg": "Compiling with torch.compile (first run will be slow)..."})
    torch.backends.cudnn.benchmark = True
    pipeline.text_encoder = torch.compile(pipeline.text_encoder, mode="max-autotune-no-cudagraphs", dynamic=False)
    pipeline.generator    = torch.compile(pipeline.generator,    mode="max-autotune-no-cudagraphs", dynamic=False)
    pipeline.vae.decoder  = torch.compile(pipeline.vae.decoder,  mode="max-autotune-no-cudagraphs", dynamic=False)

# ─────────────────────────────────────────────
# READY — signal the app we can accept jobs
# ─────────────────────────────────────────────

emit({"type": "ready"})

# ─────────────────────────────────────────────
# JOB LOOP
# ─────────────────────────────────────────────

num_output_frames = config.image_or_video_shape[1]  # e.g. 21

for raw_line in sys.stdin:
    raw_line = raw_line.strip()
    if not raw_line:
        continue

    try:
        job = json.loads(raw_line)
    except json.JSONDecodeError as e:
        emit({"type": "error", "job_id": "unknown", "msg": f"Bad JSON: {e}"})
        continue

    job_id        = job.get("job_id", "unknown")
    prompt        = job["prompt"]
    extended      = job.get("extended_prompt", "").strip()
    output_dir    = job["output_dir"]
    num_samples   = int(job.get("num_samples", 1))
    seed          = int(job.get("seed", 0))

    try:
        os.makedirs(output_dir, exist_ok=True)
        set_seed(seed)

        effective_prompt = extended if extended else prompt
        prompts = [effective_prompt] * num_samples

        log(job_id, f"Generating: '{prompt[:80]}{'...' if len(prompt)>80 else ''}'")
        log(job_id, f"Samples: {num_samples}  |  Seed: {seed}  |  Frames: {num_output_frames}")

        sampled_noise = torch.randn(
            [num_samples, num_output_frames, 16, 60, 104],
            device=device, dtype=torch.bfloat16
        )

        log(job_id, "Running denoising...")
        if config.causal:
            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
                initial_latent=None,
                low_memory=low_memory,
            )
        else:
            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
            )

        log(job_id, "Decoding video...")
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video_out = 255.0 * current_video

        pipeline.vae.model.clear_cache()

        log(job_id, "Saving video(s)...")
        saved_paths = []
        model_tag = "ema" if args.use_ema else "base"
        for i in range(num_samples):
            out_path = os.path.join(output_dir, f"0-{i}_{model_tag}.mp4")
            write_video(out_path, video_out[i], fps=16)
            saved_paths.append(out_path)
            log(job_id, f"Saved: {out_path}")

        emit({"type": "done", "job_id": job_id, "videos": saved_paths})

    except Exception as e:
        import traceback
        emit({"type": "error", "job_id": job_id, "msg": traceback.format_exc()})

# Server exits when stdin closes (parent process died)
emit({"type": "log", "job_id": "__exit__", "msg": "Server stdin closed, shutting down."})
