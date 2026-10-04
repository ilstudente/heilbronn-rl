"""Qwen3.8-27B-FP8 behind an OpenAI-compatible vLLM server on Modal, for the Hull Heilbronn Lab's Qwen agents.

Deploy:
    QWEN_VLLM_KEY=<key> modal deploy qwen_serve.py

One H100 (the FP8 weights are ~31 GB; fp8 KV cache), at most one container, scales to zero after SCALEDOWN idle
seconds, so it only costs while agents are talking to it (plus a cold start of a few minutes). Requests need the
the API key in QWEN_VLLM_KEY, attached at deploy time.
"""
import os
import subprocess
import sys
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3.8-27B-FP8"
SERVED = "qwen"                     # model name the agents use
GPU, MAX_LEN, SCALEDOWN = "H100", 131072, 300
PORT = 8000

KEY = {}
if modal.is_local():                # `modal deploy` imports this file locally; containers skip the check
    KEY = {"VLLM_API_KEY": os.environ.get("QWEN_VLLM_KEY", "")}
    if not KEY["VLLM_API_KEY"]: raise SystemExit("no Qwen key: set QWEN_VLLM_KEY")

app = modal.App("hull-qwen-serve")
image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
         .uv_pip_install("vllm==0.29.0", "huggingface_hub[hf_xet]")
         .env({"HF_HUB_CACHE": "/cache", "HF_XET_HIGH_PERFORMANCE": "1"}))
cache = modal.Volume.from_name("hull-qwen-hf-cache", create_if_missing=True)


@app.function(image=image, gpu=GPU, cpu=8, memory=32 * 1024, max_containers=1, scaledown_window=SCALEDOWN,
              timeout=24 * 60 * 60, volumes={"/cache": cache}, secrets=[modal.Secret.from_dict(KEY)])
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    import threading
    # max-num-seqs: the default 1024 exceeds the hybrid model's Mamba cache blocks on one H100 (815) and vLLM refuses
    # to start; 64 is plenty for the lab's three agents.
    proc = subprocess.Popen(["vllm", "serve", MODEL, "--served-model-name", SERVED, "--host", "0.0.0.0", "--port", str(PORT),
                             "--max-model-len", str(MAX_LEN), "--max-num-seqs", "64", "--kv-cache-dtype", "fp8",
                             "--gpu-memory-utilization", "0.92", "--reasoning-parser", "qwen3",
                             "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
                             "--api-key", os.environ["VLLM_API_KEY"]])
    # If vLLM dies, end the container at once instead of holding the GPU until the startup timeout.
    threading.Thread(target=lambda: os._exit(proc.wait() or 1), daemon=True).start()
