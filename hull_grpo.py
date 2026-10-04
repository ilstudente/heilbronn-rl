"""GRPO (LoRA) for Qwen3.8-27B on random axis-hull Heilbronn instances, on ONE H200, with a hard dollar cap.

    modal run --detach hull_grpo.py            # full run (budget-capped)
    modal run --detach hull_grpo.py --pilot    # one training step, no evals

Everything runs in one container on one GPU, time-shared:
  sample   vLLM (bf16 base + the current LoRA) serves 4 instances x 8 rollouts at once. Each rollout is the Hull Lab task
           (hull_heilbronn.py): a bash/write_file tool loop, `python3 attempt.py FILE.json` graded exactly by this
           process (never by the agent), `minutes` of wall clock and `attempts` exact grades. Token in / token out: the
           trainer sees exactly the tokens and sampler log-probs of the rollout (no re-tokenization).
  train    vLLM sleeps (weights to host RAM, KV cache freed); the trainer process moves its bf16 model + LoRA to the
           GPU, takes one TIS-weighted policy-gradient step on the groups with signal, saves the adapter, moves back.
  eval     classic Heilbronn square, k=16 (record 7/341): eval_n rollouts with the base model before training and with
           the final adapter after, eval_minutes / eval_attempts each.
Rewards/advantages: hull_heilbronn.episode_terms / group_rewards (best + capped leximin tie-break, improvement curve
over the attempt slots, penalties for invalid/duplicate attempts; group-relative, near-ties -> 0).

Budget: the run tracks its own spend (container time x RATE_PER_H) and stops training when the next step plus the
final eval would pass budget_usd. The container's timeout is a second, hard stop. Results: volume `hull-grpo`, runs/<name>/.
"""
import asyncio
import json
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3.8-27B"
RATE_PER_H = 4.54 + 32 * 0.0000131 * 3600 + 256 * 0.00000222 * 3600   # H200 + 32 cores + 256 GiB, Modal list prices
CMD_MEM_KB = 6 * 1024 * 1024                                          # per agent command: a runaway search dies alone
K_RANGE = {2: (8, 40), 3: (12, 30), 4: (16, 22)}                      # C(k, n+1) <= ~27k: exact grading stays fast
CFG = dict(steps=10, instances=4, group=8, minutes=7, attempts=5, eval_n=16, eval_minutes=5, eval_attempts=5,
           lr=4e-5, lora_r=16, max_grad_norm=1.0, budget_usd=9.5, max_turn_tokens=8192, max_context=32768,
           nudges=3, seed=0, p_n2=0.5, resume="", base_eval=True,
           rft_lambda=0.5, rft_top=2, square_frac=0.25)   # RFT term: +lambda on each group's top-2 valid rollouts   # resume: volume path of an adapter dir to continue from
TIS_CLIP, OFFLOAD_ABOVE = 3.0, 32768

app = modal.App("hull-grpo")
vol = modal.Volume.from_name("hull-grpo", create_if_missing=True)
hf_cache = modal.Volume.from_name("hull-grpo-hf-cache", create_if_missing=True)
HERE = Path(__file__).resolve().parent
image = (modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu22.04", add_python="3.12")
         .entrypoint([])
         .apt_install("git", "build-essential", "ninja-build", "zsh")
         .uv_pip_install("vllm==0.29.0", "peft>=0.18", "accelerate", "flash-linear-attention", "httpx", "numpy",
                         "scipy", "sympy", "huggingface_hub[hf_xet]", "packaging", "ninja", "wheel", "setuptools")
         .run_commands("CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='9.0' CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 "
                       "pip install --no-build-isolation causal-conv1d")
         .env({"HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false",
               "VLLM_SERVER_DEV_MODE": "1", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"})
         .add_local_file(HERE / "hull_heilbronn.py", "/root/hull_heilbronn.py"))

@app.function(image=modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub[hf_xet]")
              .env({"HF_XET_HIGH_PERFORMANCE": "1"}), cpu=8, memory=16 * 1024, timeout=60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache})
def prefetch():
    """Download the bf16 weights into the cache volume on a CPU container, so the GPU doesn't wait for them."""
    from huggingface_hub import snapshot_download
    t0 = time.time(); snapshot_download(MODEL); hf_cache.commit()
    return round(time.time() - t0)


# ======================================================================================================================
# Rollout environment (pure Python; tested locally with a fake generator)
# ======================================================================================================================
TOOLS = [
    dict(type='function', function=dict(
        name='bash', description='Run a shell command (zsh) in the working directory and return its output.',
        parameters=dict(type='object', required=['command'], properties=dict(
            command=dict(type='string'),
            timeout=dict(type='integer', description='Seconds before the command is killed (max 150, default 120).'))))),
    dict(type='function', function=dict(
        name='write_file', description='Write text to a file in the working directory (creates or overwrites).',
        parameters=dict(type='object', required=['path', 'content'], properties=dict(
            path=dict(type='string'), content=dict(type='string'))))),
]
STRING_PARAMS = {'bash': {'command'}, 'write_file': {'path', 'content'}}
SYSTEM = ('You are an autonomous agent working in a sandboxed directory. Use the tools to run code and inspect '
          'results. Keep going until the task is done, then give a short final answer without a tool call.')
IM_END = "<|im_end|>"
TEMPLATE_KWARGS = {"enable_thinking": True, "reasoning_effort": "medium"}
ATTEMPT_PY = r'''"""python3 attempt.py candidate.json  -> exact grade; uses one of your attempts"""
import json, sys, time, uuid
from pathlib import Path
here = Path(__file__).resolve().parent
text = Path(sys.argv[1]).read_text()
if len(text) > 65536: sys.exit('candidate too large')
rid = uuid.uuid4().hex
(here / 'outbox').mkdir(exist_ok=True); (here / 'inbox').mkdir(exist_ok=True)
tmp = here / 'outbox' / (rid + '.tmp'); tmp.write_text(text); tmp.replace(tmp.with_suffix('.req'))
for _ in range(1800):
    out = here / 'inbox' / (rid + '.json')
    if out.exists(): print(out.read_text()); break
    time.sleep(0.2)
else: sys.exit('no answer from the grader')
'''
TIMEOUT_SH = r'''#!/usr/bin/env python3
import os, signal, subprocess, sys
limit = float(sys.argv[1].rstrip('s')); p = subprocess.Popen(sys.argv[2:], start_new_session=True)
try: sys.exit(p.wait(timeout=limit))
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGTERM); sys.exit(124)
'''


def task_text(H, inst, minutes, attempts):
    return (H.prompt(inst) + f"""

Your working directory holds instance.json (this instance) and attempt.py.
- `python3 attempt.py FILE.json` gives an exact grade. You have {attempts} attempts in total, and your result is your
  BEST attempt. Each answer also reports how many attempts you have left.
- Submit a first attempt early, within the first third of your time: an episode without a valid attempt scores
  worst of all, worse than an invalid attempt.
- There is no exact grader in your directory. Compute areas yourself in floating point as much as you like;
  attempts are the only exact grades, so use them on candidates you expect to be good.
- You have {minutes} minutes. Keep single commands under ~2 minutes (`timeout SECONDS cmd` works here), prefix
  python with OMP_NUM_THREADS=1, and don't use the network. NumPy and SciPy are installed.
- Exact grading: points on the boundary must satisfy the constraints exactly, so round inward or use fractions.
When you stop, give a one-line summary of your method and your best attempt score.""")


def to_template(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m["role"] == "assistant":
            m["reasoning_content"] = m.pop("reasoning", "") or ""
            if m.get("tool_calls"):
                m["tool_calls"] = [{**c, "function": {"name": c["function"]["name"],
                                                      "arguments": json.loads(c["function"]["arguments"] or "{}")}}
                                   for c in m["tool_calls"]]
        out.append(m)
    return out


def render(tok, msgs, gen):
    return tok.apply_chat_template(to_template(msgs), tools=TOOLS, tokenize=False, add_generation_prompt=gen,
                                   **TEMPLATE_KWARGS)


def parse_turn(text):
    text = text.replace(IM_END, "")
    reasoning, rest = text.split("</think>", 1) if "</think>" in text else (text, "")
    calls = []
    for m in re.finditer(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", rest, re.S):
        name, args = m.group(1), {}
        for p in re.finditer(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", m.group(2), re.S):
            k, v = p.group(1), p.group(2)
            if k in STRING_PARAMS.get(name, ()):
                args[k] = v
            else:
                try: args[k] = json.loads(v.strip())
                except json.JSONDecodeError: args[k] = v.strip()
        calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "name": name, "arguments": json.dumps(args)})
    content = re.sub(r"<tool_call>.*?(</tool_call>|$)", "", rest, flags=re.S).strip()
    return reasoning.strip(), content, calls


def canonical(points):
    from fractions import Fraction
    return sorted(tuple(round(float(Fraction(str(c))), 6) for c in p) for p in points)


class Grader:
    """Serves one rollout's attempt.py requests (outbox -> inbox), enforcing its budget and deadline."""

    def __init__(self, H, inst, work, attempts, deadline, pool):
        self.H, self.inst, self.work, self.attempts, self.deadline, self.pool = H, inst, work, attempts, deadline, pool
        self.records, self.t0 = [], time.time()

    def _grade(self, text):
        if time.time() > self.deadline: return dict(error='time is up')
        if len(self.records) >= self.attempts: return dict(error=f'no attempts left ({self.attempts} used)')
        r = self.H.grade(self.inst, text[:65536])
        rec = dict(valid=r['valid'], score=r['score'], score_exact=r.get('score_exact'), error=r.get('error'),
                   smallest=r.get('smallest'), t=round(time.time() - self.t0, 1))
        if r['valid']:
            key = canonical(json.loads(text))
            same = next((j + 1 for j, o in enumerate(self.records) if o.get('key') == key), None)
            rec['key'] = key
            if same: rec['dup'] = f'same layout as your attempt {same}'
        self.records.append(rec)
        return dict(valid=r['valid'], score=r['score'], error=r.get('error'), dup=rec.get('dup'),
                    attempts_left=self.attempts - len(self.records), seconds_left=round(self.deadline - time.time()))

    async def serve(self):
        loop = asyncio.get_running_loop()
        box, inbox = self.work / 'outbox', self.work / 'inbox'
        while True:
            for f in sorted(box.glob('*.req')) if box.exists() else []:
                text = f.read_text(); f.unlink()
                out = await loop.run_in_executor(self.pool, self._grade, text)
                inbox.mkdir(exist_ok=True); (inbox / (f.stem + '.json')).write_text(json.dumps(out, indent=1))
            await asyncio.sleep(0.3)

    def footer(self):
        left = max(0, int(self.deadline - time.time()))
        best = max((r['score'] for r in self.records if r['valid']), default=None)
        return (f"\n[{left // 60}m{left % 60:02d}s left · {self.attempts - len(self.records)} attempts left · "
                + (f"your best: {best:.6f}]" if best is not None else "your best: none yet, submit an attempt]"))


async def run_tool(call, work, deadline, env):
    try: a = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError: return 'error: arguments are not valid JSON'
    if call["name"] == 'write_file':
        path = Path(str(a.get('path', '')))
        if not path.parts or path.is_absolute() or '..' in path.parts: return 'error: path must stay inside the working directory'
        (work / path).parent.mkdir(parents=True, exist_ok=True); (work / path).write_text(str(a.get('content', '')))
        return f'wrote {len(str(a.get("content", "")))} characters to {path}'
    if call["name"] == 'bash':
        try: req = int(a.get('timeout') or 120)
        except (TypeError, ValueError): req = 120
        limit = max(1, min(150, req, int(deadline - time.time()) + 5))
        cmd = f"ulimit -v {CMD_MEM_KB} 2>/dev/null; " + str(a.get('command', ''))
        p = await asyncio.create_subprocess_exec('zsh', '-c', cmd, cwd=work, env=env,
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                 start_new_session=True)
        try:
            out, _ = await asyncio.wait_for(p.communicate(), timeout=limit)
        except asyncio.TimeoutError:
            try: os.killpg(p.pid, 9)
            except ProcessLookupError: pass
            return f'error: command timed out after {limit}s'
        out = out.decode(errors='replace').strip() or '(no output)'
        return (out if len(out) <= 8000 else out[:3000] + '\n…[truncated]…\n' + out[-4000:]) + f'\n[exit {p.returncode}]'
    return f'error: unknown tool {call["name"]}'


async def rollout(H, tok, generate, inst, minutes, attempts, work, pool, max_turn_tokens=8192, max_context=65536,
                  nudges=3):
    """One episode. Returns ids/mask/lps (token in / token out), the grader's records and stats."""
    work.mkdir(parents=True, exist_ok=True)
    (work / 'instance.json').write_text(json.dumps(inst, indent=2)); (work / 'attempt.py').write_text(ATTEMPT_PY)
    (work / 'bin').mkdir(exist_ok=True); (work / 'bin' / 'timeout').write_text(TIMEOUT_SH)
    (work / 'bin' / 'timeout').chmod(0o755)
    env = {'PATH': f"{work / 'bin'}:{os.environ.get('PATH', '')}", 'HOME': str(work), 'LANG': 'C.UTF-8',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    t0 = time.time(); deadline = t0 + 60 * minutes
    grader = Grader(H, inst, work, attempts, deadline, pool)
    server = asyncio.ensure_future(grader.serve())
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task_text(H, inst, minutes, attempts)}]
    ids = tok(render(tok, msgs, True), add_special_tokens=False)["input_ids"]
    mask, lps = [0] * len(ids), [0.0] * len(ids)
    end = tok.convert_tokens_to_ids(IM_END)
    status, turns, used_nudges, tool_calls, gen_tokens = 'finished', 0, 0, 0, 0

    def append(new_msgs, upto):
        """Append the rendering of new (non-generated) messages + the next generation prompt."""
        msgs.extend(new_msgs)
        full = render(tok, msgs, True)
        if not (upto.endswith(IM_END + "\n") and full.startswith(upto)):
            return False
        suffix = tok(full[len(upto) - 1:], add_special_tokens=False)["input_ids"]
        ids.extend(suffix); mask.extend([0] * len(suffix)); lps.extend([0.0] * len(suffix))
        return True
    try:
        while True:
            left = deadline - time.time()
            if left <= 1: status = 'time'; break
            if len(ids) >= max_context - 512: status = 'context'; break
            try:
                gen_ids, gen_lps = await asyncio.wait_for(
                    generate(ids, min(max_turn_tokens, max_context - len(ids) - 256)), timeout=left)
            except asyncio.TimeoutError:
                status = 'time'; break                          # the unfinished turn is dropped (not trained)
            turns += 1; gen_tokens += len(gen_ids)
            ids += gen_ids; mask += [1] * len(gen_ids); lps += gen_lps
            if not gen_ids or gen_ids[-1] != end:
                status = 'length'; break                        # cut off mid-turn: kept, the episode ends
            reasoning, content, calls = parse_turn(tok.decode(gen_ids, skip_special_tokens=False))
            amsg = {"role": "assistant", "content": content, "reasoning": reasoning}
            if calls:
                amsg["tool_calls"] = [{"id": c["id"], "type": "function",
                                       "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
            upto = render(tok, msgs + [amsg], False)
            msgs.append(amsg)
            if not calls:
                left = deadline - time.time()
                if used_nudges >= nudges or len(grader.records) >= attempts or left < 45:
                    break
                used_nudges += 1                                # like the lab's resume: keep going while budget left
                ok = append([{"role": "user", "content": f"{int(left)} seconds and {attempts - len(grader.records)} "
                              "attempts left. Keep improving: try an idea different from what you did so far, and "
                              "spend attempts on candidates you expect to beat your best."}], upto)
                if not ok: status = 'render_mismatch'; break
                continue
            outs = []
            for c in calls:
                tool_calls += 1
                outs.append(await run_tool(c, work, deadline, env) + grader.footer())
            if not append([{"role": "tool", "tool_call_id": c["id"], "content": o} for c, o in zip(calls, outs)], upto):
                status = 'render_mismatch'; break
    except Exception as e:                                        # noqa: BLE001 - one bad rollout must not end a step
        status = f'error: {type(e).__name__}: {e}'
    await asyncio.sleep(0.8)                                      # let a just-written attempt be served
    server.cancel()
    return dict(ids=ids, mask=mask, lps=lps, records=grader.records, status=status, turns=turns, tool_calls=tool_calls,
                gen_tokens=gen_tokens, seconds=round(time.time() - t0, 1), messages=msgs)


SQUARE_KS = [k for k in range(12, 21) if k != 16]                     # unit squares near the eval, never k=16 itself


def sample_instance(H, rng, p_n2=1 / 3, square_frac=0.0):
    if rng.random() < square_frac:
        return H.square_instance(rng.choice(SQUARE_KS))
    n = 2 if rng.random() < p_n2 else rng.choice([3, 4])
    return H.make_instance(n, rng.randint(*K_RANGE[n]), rng.randrange(10**9))


# ======================================================================================================================
# Trainer process (single GPU; model parked in host RAM between steps)
# ======================================================================================================================
def _trainer(inq, outq):
    import traceback
    try:
        _trainer_main(inq, outq)
    except BaseException:                                         # report instead of dying silently
        outq.put(("error", traceback.format_exc()[-4000:]))
        raise


def _trainer_main(inq, outq):
    import contextlib
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForImageTextToText

    cfg = inq.get()
    base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": "cpu"},
                                                       attn_implementation="sdpa")
    base.config.use_cache = False
    torch.manual_seed(0)
    if cfg.get("resume"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, cfg["resume"], is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=cfg["lora_r"], lora_alpha=2 * cfg["lora_r"], lora_dropout=0.0, bias="none",
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)
    if cfg.get("resume") and os.path.exists(f"{cfg['resume']}/optimizer.pt"):
        opt.load_state_dict(torch.load(f"{cfg['resume']}/optimizer.pt", map_location="cpu"))

    def move(dev):
        model.to(dev)
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v): st[k] = v.to(dev)
        if dev == "cpu": torch.cuda.empty_cache()
    outq.put(("ready", sum(p.numel() for p in trainable)))
    while True:
        cmd, arg = inq.get()
        if cmd != "step":
            break
        samples, path = arg
        t0 = time.time(); move("cuda"); t_move = time.time() - t0
        total = max(1.0, float(sum(int(s["mask"][1:].sum()) for s in samples)))
        lm = model.base_model.model
        model.train()
        loss_sum = w_sum = w_n = clipped = 0.0
        for s in samples:
            with torch.autograd.graph.save_on_cpu(pin_memory=True) if len(s["ids"]) > OFFLOAD_ABOVE else contextlib.nullcontext():
                x = torch.from_numpy(s["ids"].astype("int64"))[None].cuda()
                h = lm.model(input_ids=x, use_cache=False).last_hidden_state[0]
                pos = torch.from_numpy(s["mask"][1:]).cuda().nonzero().squeeze(-1)
                tgt = x[0, pos + 1]
                vlp = torch.from_numpy(s["lps"][1:]).cuda()[pos]
                acc = torch.zeros((), device="cuda", dtype=torch.float32)
                for c in range(0, len(pos), 2048):
                    p, t, v = pos[c:c + 2048], tgt[c:c + 2048], vlp[c:c + 2048]

                    def chunk(hh, tt, vv):
                        lp = torch.log_softmax(lm.lm_head(hh).float(), -1).gather(-1, tt[:, None]).squeeze(-1)
                        w = torch.exp(lp.detach() - vv).clamp(max=TIS_CLIP)
                        w = torch.where(torch.isnan(vv), torch.ones_like(w), w)
                        return (w * lp).sum(), w.sum(), (w >= TIS_CLIP).sum()
                    wl, ws, wc = checkpoint(chunk, h[p], t, v, use_reentrant=False)
                    acc = acc + wl; w_sum += ws.item(); w_n += len(p); clipped += wc.item()
                loss = -(s["advantage"] * acc) / total
            loss.backward(); loss_sum += loss.item()
            del h, x
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"]).item()
        opt.step(); opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 2**30
        model.save_pretrained(path)
        torch.save(opt.state_dict(), f"{path}/optimizer.pt")     # lets a later run resume exactly
        c = json.load(open(f"{path}/adapter_config.json")); c["base_model_name_or_path"] = MODEL
        json.dump(c, open(f"{path}/adapter_config.json", "w"))
        t1 = time.time(); move("cpu"); torch.cuda.reset_peak_memory_stats()
        outq.put(("step", dict(loss=loss_sum, trained=len(samples), tokens=total, grad_norm=gnorm,
                               iw_mean=w_sum / max(1.0, w_n), iw_clip=clipped / max(1.0, w_n), peak_gib=round(peak, 1),
                               train_seconds=round(t1 - t0, 1), move_seconds=round(t_move + time.time() - t1, 1))))


# ======================================================================================================================
# The run
# ======================================================================================================================
@app.function(image=image, gpu="H200", cpu=32, memory=256 * 1024, timeout=16 * 60,   # hard stop: 16 min ≈ $2.15 (re-eval)
              volumes={"/root/.cache/huggingface": hf_cache, "/vol": vol})
def train(cfg: dict, name: str, pilot: bool = False):
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    import numpy as np
    import torch.multiprocessing as mp
    from transformers import AutoTokenizer
    sys.path.insert(0, "/root")
    import hull_heilbronn as H

    t_start = time.time()
    spend = lambda: (time.time() - t_start) / 3600 * RATE_PER_H
    run_dir = Path("/vol/runs") / name; run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dict(cfg, pilot=pilot, model=MODEL, rate_per_h=RATE_PER_H)))

    def log(**kv):
        kv = {"t": round(time.time() - t_start, 1), "spend_usd": round(spend(), 2), **kv}
        with (run_dir / "metrics.jsonl").open("a") as f: f.write(json.dumps(kv, default=str) + "\n")
        vol.commit(); print(json.dumps(kv, default=str), flush=True)

    vllm = subprocess.Popen(["vllm", "serve", MODEL, "--host", "127.0.0.1", "--port", "8000",
                             "--max-model-len", str(cfg["max_context"]), "--max-num-seqs", "64",
                             "--gpu-memory-utilization", "0.85", "--enable-sleep-mode", "--enable-prefix-caching",
                             "--enable-lora", "--max-lora-rank", str(cfg["lora_r"]), "--max-loras", "2",
                             "--limit-mm-per-prompt", '{"image": 0, "video": 0}', "--disable-uvicorn-access-log"])
    ctx = mp.get_context("spawn")
    inq, outq = ctx.Queue(), ctx.Queue()
    trainer = ctx.Process(target=_trainer, args=(inq, outq), daemon=True)
    trainer.start(); inq.put(cfg)                                # loads the model into host RAM while vLLM starts
    tok = AutoTokenizer.from_pretrained(MODEL)

    def wait_trainer(kind, timeout):
        """Next trainer message; if the trainer dies or errors, log it and end the run at once (no idle GPU)."""
        import queue
        t_end = time.time() + timeout
        while time.time() < t_end:
            try:
                k, v = outq.get(timeout=10)
            except queue.Empty:
                if not trainer.is_alive():
                    log(event="trainer_died", exitcode=trainer.exitcode)
                    raise RuntimeError(f"trainer exited with code {trainer.exitcode}")
                continue
            if k == "error":
                log(event="trainer_error", traceback=v)
                raise RuntimeError("trainer failed:\n" + v)
            return k, v
        log(event="trainer_timeout", waiting_for=kind)
        raise RuntimeError(f"trainer sent no {kind} in {timeout}s")
    while True:
        try:
            if httpx.get("http://127.0.0.1:8000/health", timeout=5).status_code == 200: break
        except Exception: pass
        if vllm.poll() is not None: raise RuntimeError("vllm serve exited during startup")
        time.sleep(5)
    log(event="vllm_ready")
    kind, n_params = wait_trainer("ready", 30 * 60)
    log(event="trainer_ready", trainable_params=n_params)
    pool = ThreadPoolExecutor(64)
    rng = random.Random(cfg["seed"])
    end = tok.convert_tokens_to_ids(IM_END)

    async def run_many(jobs, adapter, tag):
        async with httpx.AsyncClient(timeout=httpx.Timeout(3600, connect=30), limits=httpx.Limits(max_connections=128)) as cl:
            def make_generate(model):
                async def generate(ids, max_tokens):
                    return await gen(model, ids, max_tokens)
                return generate

            async def gen(model, ids, max_tokens):
                body = {"model": model, "prompt": ids, "max_tokens": max(1, max_tokens), "temperature": 1.0,
                        "top_p": 0.95, "top_k": 20, "logprobs": 0, "return_tokens_as_token_ids": True,
                        "skip_special_tokens": False, "stop_token_ids": [end]}
                r = await cl.post("http://127.0.0.1:8000/v1/completions", json=body); r.raise_for_status()
                ch = r.json()["choices"][0]
                toks = [int(x.split(":", 1)[1]) for x in ch["logprobs"]["tokens"]]
                lp = [float("nan") if x is None else float(x) for x in ch["logprobs"]["token_logprobs"]]
                if ch["finish_reason"] == "stop" and (not toks or toks[-1] != end): toks.append(end); lp.append(float("nan"))
                return toks, lp
            outs = await asyncio.gather(*(rollout(H, tok, make_generate(j.get("adapter", adapter) or MODEL),
                                                  j["inst"], j["minutes"], j["attempts"],
                                                  Path(f"/tmp/work/{tag}/{i}"), pool, cfg["max_turn_tokens"],
                                                  cfg["max_context"], cfg["nudges"]) for i, j in enumerate(jobs)))
        for j, o in zip(jobs, outs):
            o["terms"] = H.episode_terms(o["records"], budget=j["attempts"])
        (run_dir / "rollouts").mkdir(exist_ok=True)
        (run_dir / "rollouts" / f"{tag}.json").write_text(json.dumps([dict(
            instance={k: j["inst"][k] for k in ("n", "k", "seed")}, status=o["status"], turns=o["turns"],
            tool_calls=o["tool_calls"], gen_tokens=o["gen_tokens"], seconds=o["seconds"], tokens=len(o["ids"]),
            terms=o["terms"], attempts=[{k: v for k, v in r.items() if k not in ("smallest", "key")} for r in o["records"]])
            for j, o in zip(jobs, outs)], default=str))
        tdir = run_dir / "transcripts" / tag; tdir.mkdir(parents=True, exist_ok=True)   # for later RFT / inspection
        for i, (j, o) in enumerate(zip(jobs, outs)):
            (tdir / f"{i:03d}.json").write_text(json.dumps(dict(instance=j["inst"], terms=o["terms"], status=o["status"],
                                                               messages=o["messages"]), default=str))
            np.savez_compressed(tdir / f"{i:03d}.npz", ids=np.asarray(o["ids"], np.int32),
                                mask=np.asarray(o["mask"], np.bool_), lps=np.asarray(o["lps"], np.float32))
        vol.commit()
        return outs

    def eval_jobs():
        return [dict(inst=H.square_instance(16), minutes=cfg["eval_minutes"], attempts=cfg["eval_attempts"], g=-1)
                for _ in range(cfg["eval_n"])]

    def evaluate(adapter, tag, outs=None):
        rec = 7 / 341
        outs = outs or asyncio.run(run_many(eval_jobs(), adapter, tag))
        pct = sorted(100 * o["terms"]["best"] / rec for o in outs)
        log(event="eval", tag=tag, adapter=adapter, pct_of_record=[round(p, 1) for p in pct],
            mean=round(sum(pct) / len(pct), 1), median=round(pct[len(pct) // 2], 1), best=round(pct[-1], 1),
            invalid=sum(o["terms"]["invalid"] for o in outs), statuses=[o["status"] for o in outs])

    def sleep_vllm(on):
        r = httpx.post(f"http://127.0.0.1:8000/{'sleep?level=1' if on else 'wake_up'}", timeout=600)
        r.raise_for_status()

    with_eval = not pilot and cfg["eval_n"] > 0
    adapter, step_secs = None, None
    if cfg.get("resume"):                              # continue from an earlier run's adapter
        r = httpx.post("http://127.0.0.1:8000/v1/load_lora_adapter", json={"lora_name": "resume",
                       "lora_path": cfg["resume"]}, timeout=600)
        if r.status_code != 200: raise RuntimeError(f"loading resume adapter: {r.status_code} {r.text[:300]}")
        adapter = "resume"
    if cfg.get("eval_only"):                           # re-eval: base and adapter in ONE wave (equal CPU share)
        both = [dict(j, adapter=None) for j in eval_jobs()] + [dict(j, adapter=adapter) for j in eval_jobs()]
        outs = asyncio.run(run_many(both, None, "reeval"))
        evaluate(None, "reeval_base", outs[:cfg["eval_n"]])
        evaluate(adapter, "reeval_trained", outs[cfg["eval_n"]:])
        inq.put(("stop", None)); log(event="done", final_adapter=adapter); return
    if with_eval and cfg.get("base_eval", True):     # base model, alone: same conditions as the final eval
        evaluate(None, "eval_base")
    for step in range(1, (1 if pilot else cfg["steps"]) + 1):
        est_step = (step_secs or 15 * 60) / 3600 * RATE_PER_H
        est_eval = ((cfg["eval_minutes"] + 1) / 60 * RATE_PER_H) if with_eval else 0.0
        if spend() + est_step + est_eval > cfg["budget_usd"]:
            log(event="budget_stop", step=step, est_step_usd=round(est_step, 2), est_eval_usd=round(est_eval, 2)); break
        t0 = time.time()
        insts = [sample_instance(H, rng, cfg["p_n2"], cfg.get("square_frac", 0.0)) for _ in range(cfg["instances"])]
        jobs = [dict(inst=i, minutes=cfg["minutes"], attempts=cfg["attempts"], g=g) for g, i in enumerate(insts)
                for _ in range(cfg["group"])]
        outs = asyncio.run(run_many(jobs, adapter, f"step{step:03d}"))
        t_roll = time.time() - t0
        samples, groups_with_signal, rft_kept = [], 0, 0
        for g in range(len(insts)):
            members = [o for j, o in zip(jobs, outs) if j["g"] == g]
            rewards = H.group_rewards([o["terms"] for o in members])
            if any(r["advantage"] for r in rewards): groups_with_signal += 1
            # RFT term: the group's best rollouts (valid, non-zero) also get +rft_lambda, a plain imitation push
            ranked = sorted(range(len(members)), key=lambda i: -members[i]["terms"]["final"])
            top = {i for i in ranked[:cfg.get("rft_top", 0)] if members[i]["terms"]["best"] > 0}
            rft_kept += len(top)
            for i, (o, r) in enumerate(zip(members, rewards)):
                w = float(r["advantage"]) + (cfg.get("rft_lambda", 0.0) if i in top else 0.0)
                if w and any(o["mask"]):
                    samples.append(dict(ids=np.asarray(o["ids"], np.int32), mask=np.asarray(o["mask"], np.bool_),
                                        lps=np.asarray(o["lps"], np.float32), advantage=w))
        best_ratio = [o["terms"]["best"] for o in outs]
        roll = dict(instances=[(i["n"], i["k"]) for i in insts], rollout_seconds=round(t_roll, 1),
                    mean_tokens=round(sum(len(o["ids"]) for o in outs) / len(outs)),
                    max_tokens=max(len(o["ids"]) for o in outs),
                    mean_gen_tokens=round(sum(o["gen_tokens"] for o in outs) / len(outs)),
                    mean_attempts=round(sum(len(o["records"]) for o in outs) / len(outs), 2),
                    invalid=sum(o["terms"]["invalid"] for o in outs), no_attempt=sum(not o["records"] for o in outs),
                    statuses={s: sum(o["status"] == s for o in outs) for s in {o["status"] for o in outs}},
                    groups_with_signal=groups_with_signal, samples=len(samples), rft_kept=rft_kept,
                    mean_reward_best=round(sum(best_ratio) / len(best_ratio), 6))
        if not samples:
            log(event="step", step=step, trained=0, **roll); step_secs = time.time() - t0; continue
        sleep_vllm(True)
        path = str(run_dir / "adapters" / f"step{step:03d}")
        inq.put(("step", (samples, path)))
        kind, tr = wait_trainer("step", 60 * 60)
        vol.commit()
        sleep_vllm(False)
        name_ = f"step{step:03d}"
        r = httpx.post("http://127.0.0.1:8000/v1/load_lora_adapter", json={"lora_name": name_, "lora_path": path},
                       timeout=600)
        if r.status_code != 200: raise RuntimeError(f"loading {name_}: {r.status_code} {r.text[:300]}")
        adapter = name_
        step_secs = time.time() - t0
        log(event="step", step=step, adapter=adapter, step_seconds=round(step_secs, 1), **roll, **tr)
    if with_eval and adapter and cfg.get("final_both"):   # base and trained side by side in one wave (fair)
        both = [dict(j, adapter=None) for j in eval_jobs()] + [dict(j, adapter=adapter) for j in eval_jobs()]
        outs = asyncio.run(run_many(both, None, "final_both"))
        evaluate(None, "final_base", outs[:cfg["eval_n"]])
        evaluate(adapter, "final_trained", outs[cfg["eval_n"]:])
    elif with_eval and adapter:
        evaluate(adapter, "eval_trained")
    inq.put(("stop", None))
    log(event="done", final_adapter=adapter)


@app.function(image=image, cpu=16, memory=192 * 1024, timeout=30 * 60,
              volumes={"/root/.cache/huggingface": hf_cache})
def diag_trainer():
    """CPU only: start the trainer process as train() does and report whether it loads (or how it dies)."""
    import queue
    import resource
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    inq, outq = ctx.Queue(), ctx.Queue()
    t0 = time.time()
    proc = ctx.Process(target=_trainer, args=(inq, outq), daemon=True)
    proc.start(); inq.put(CFG)
    while time.time() - t0 < 25 * 60:
        try:
            k, v = outq.get(timeout=10)
            inq.put(("stop", None)); proc.join(60)
            out = dict(result=k, detail=v, seconds=round(time.time() - t0)); print("DIAG", out, flush=True); return out
        except queue.Empty:
            if not proc.is_alive():
                out = dict(result="died", exitcode=proc.exitcode, seconds=round(time.time() - t0)); print("DIAG", out, flush=True); return out
    return dict(result="timeout", seconds=round(time.time() - t0))


@app.local_entrypoint()
def main(pilot: bool = False, name: str = "", steps: int = 0, budget_usd: float = 0.0, resume: str = "",
         no_base_eval: bool = False, eval_only: bool = False, final_both: bool = False, square_frac: float = -1.0):
    print(f"prefetch: weights cached in {prefetch.remote()}s (CPU container)", flush=True)
    cfg = dict(CFG, **({"steps": steps} if steps else {}), **({"budget_usd": budget_usd} if budget_usd else {}),
               **({"resume": resume} if resume else {}), **({"base_eval": False} if no_base_eval else {}),
               **({"eval_only": True} if eval_only else {}), **({"final_both": True} if final_both else {}),
               **({"square_frac": square_frac} if square_frac >= 0 else {}))
    name = name or time.strftime("%Y%m%d-%H%M%S") + ("-pilot" if pilot else "")
    print(f"run {name}: {json.dumps(cfg)}  (≈ ${RATE_PER_H:.2f}/h on the GPU, hard stop at 16 min)", flush=True)
    train.remote(cfg, name, pilot)
