"""Hull Heilbronn Lab: pick n, k -> random axis hull -> three headless Claude agents compete on it, live.

python3 hull_lab.py [--port 8771]      then open http://127.0.0.1:8771

Rules (per agent): ATTEMPTS exact grades, all private. Agents work in isolation and never see each other's results.
An agent's result is its best attempt.

Each agent is a `claude -p` session (your claude.ai login; no API key is passed through) in its own temp directory
holding instance.json and attempt.py. There is no grader in the directory: attempt.py drops a request in outbox/ and
the server grades it, enforces the budget and answers in inbox/. When an agent stops while it still has attempts
and time left, the server resumes the same session with its own attempt history.
Agents are Claude (`claude -p`) or Qwen (qwen_agent.py against QWEN_BASE_URL, the qwen_serve.py endpoint; key from QWEN_API_KEY,
else the repo's .secrets/qwen_vllm_key). HULL_AGENT_CMD overrides the agent command (used for testing with mock_agent.py).
"""
import argparse
import json
import os
import random
import shlex
import sys
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import hull_heilbronn as H

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'hull_runs'
AGENTS, ATTEMPTS, MAX_TURNS = 3, 10, 8
LOCK = threading.Lock()
PROCS = {}  # run id -> [Popen | None] (current process per agent)

TIMEOUT = r'''#!/usr/bin/env python3
"""timeout SECONDS COMMAND [ARGS...]: GNU-style shim (macOS has none). Exit 124 when the limit is hit."""
import os, signal, subprocess, sys
if len(sys.argv) < 3: sys.exit('usage: timeout SECONDS COMMAND [ARGS...]')
limit = float(sys.argv[1].rstrip('s'))
p = subprocess.Popen(sys.argv[2:], start_new_session=True)
try: sys.exit(p.wait(timeout=limit))
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGTERM)
    try: p.wait(timeout=5)
    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL)
    sys.exit(124)
'''

CLIENT = r'''"""python3 {name}.py candidate.json  -> {doc}"""
import json, sys, time, uuid
from pathlib import Path
here = Path(__file__).resolve().parent
text = Path(sys.argv[1]).read_text()
if len(text) > 65536: sys.exit('candidate too large')
rid = uuid.uuid4().hex
(here / 'outbox').mkdir(exist_ok=True); (here / 'inbox').mkdir(exist_ok=True)
tmp = here / 'outbox' / (rid + '.tmp'); tmp.write_text(json.dumps(dict(kind='{name}', text=text))); tmp.replace(tmp.with_suffix('.req'))
for _ in range(1800):
    out = here / 'inbox' / (rid + '.json')
    if out.exists(): print(out.read_text()); break
    time.sleep(0.2)
else: sys.exit('no answer from the grading server')
'''


def task_text(inst, minutes):
    return (H.prompt(inst) + f"""

Your working directory holds instance.json (this instance) and attempt.py.
- `python3 attempt.py FILE.json` gives an exact grade. You have {ATTEMPTS} attempts in total, and your result is
  your BEST attempt. Each answer also reports how many attempts you have left.
- Submit a first attempt early, within the first third of your time: unsubmitted work scores nothing, and
  you can keep improving afterwards.
- There is no exact grader in your directory. Compute areas yourself in floating point as much as you like;
  attempts are the only exact grades, so use them on candidates you expect to be good.
- You have {minutes} minutes. When you stop, you may be resumed with your attempt history while you still have
  attempts left, so use the time to find something better rather than stopping early.
- Keep single commands under ~2 minutes (`timeout SECONDS cmd` works here), prefix python with OMP_NUM_THREADS=1,
  and don't use the network.
- This machine runs macOS: BSD tools, so `sed -i` needs `sed -i ''`. Prefer writing whole files (e.g. with a
  heredoc or a file-writing tool) over in-place edits.
- Exact grading: points on the boundary must satisfy the constraints exactly, so round inward or use fractions.
When you stop, give a one-line summary of your method and your best attempt score.""")


def agent_env(work=None):
    """Allowlisted environment: no ANTHROPIC_* keys and no host-session CLAUDE_* variables, so `claude` uses its own
    stored claude.ai login instead of an inherited API key (which would take precedence)."""
    keep = ('PATH', 'HOME', 'USER', 'LOGNAME', 'SHELL', 'TMPDIR', 'LANG', 'TERM', 'SSL_CERT_FILE')
    env = {k: v for k, v in os.environ.items() if k in keep or k.startswith('LC_')}
    env.update({k: v for k, v in os.environ.items() if k in ('QWEN_API_KEY', 'QWEN_REASONING_EFFORT')})  # Qwen agents only
    if 'QWEN_API_KEY' not in env and os.environ.get('QWEN_VLLM_KEY'):   # default: the Qwen server key
        env['QWEN_API_KEY'] = os.environ['QWEN_VLLM_KEY']
    if work: env['PATH'] = f"{Path(work) / 'bin'}:{env.get('PATH', '')}"  # provides `timeout`
    return dict(env, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def agent_cmd(model, session=None, backend='claude'):
    if os.environ.get('HULL_AGENT_CMD'): return shlex.split(os.environ['HULL_AGENT_CMD'])
    if backend == 'qwen':
        return ([sys.executable, str(ROOT / 'qwen_agent.py'), '--base-url', os.environ['QWEN_BASE_URL'], '--model', model or 'qwen']
                + (['--resume', session] if session else []))
    cmd = ['claude', '-p', '--output-format', 'stream-json', '--verbose', '--setting-sources', 'project',
           '--strict-mcp-config', '--allowedTools', 'Bash,Read,Write,Edit,Glob,Grep',
           '--disallowedTools', 'WebFetch,WebSearch,Task,Agent']
    return cmd + (['--model', model] if model else []) + (['--resume', session] if session else [])


def launch(run, i, work, message, model, session=None, backend='claude', deadline=None):
    log = (run / f'agent{i + 1}.jsonl').open('ab')
    env = dict(agent_env(work), **({'HULL_DEADLINE': str(deadline)} if deadline else {}))  # Qwen agents show time left
    p = subprocess.Popen(agent_cmd(model, session, backend), cwd=work, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.PIPE, env=env, start_new_session=True, text=True)
    p.stdin.write(message); p.stdin.close()  # prompt via stdin: variadic tool flags would swallow a positional prompt
    return p


def start(n, k, seed, minutes, model, backend='claude', domain='hull'):
    if backend == 'qwen':
        url = os.environ.get('QWEN_BASE_URL')
        if not url: raise ValueError('Qwen endpoint not configured: start the lab with QWEN_BASE_URL (and QWEN_API_KEY) set')
        if 'modal.run' in url:      # a request can scale the Modal server up from zero: that is a GPU launch
            pass
    if seed is None: seed = random.randrange(10**6)
    inst = H.square_instance(k) if domain == 'square' else H.make_instance(n, k, seed)
    rid = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:4]
    run = RUNS / rid; run.mkdir(parents=True)
    ref = H.baseline(inst, 200)
    text = task_text(inst, minutes)
    meta = dict(id=rid, instance=inst, prompt=text, minutes=minutes, model=model or 'CLI default', started=time.time(),
                deadline=time.time() + 60 * minutes, reference=str(ref), reference_score=float(ref),
                rules=dict(attempts=ATTEMPTS, max_turns=MAX_TURNS), backend=backend,
                command=' '.join(agent_cmd(model, backend=backend)), agents=[])
    procs = []
    for i in range(AGENTS):
        work = Path(tempfile.mkdtemp(prefix=f'hull-{rid}-a{i + 1}-'))
        (work / 'instance.json').write_text(json.dumps(inst, indent=2))
        (work / 'attempt.py').write_text(CLIENT.replace('{name}', 'attempt').replace('{doc}', 'exact grade; uses one of your attempts'))
        (work / 'bin').mkdir(); (work / 'bin' / 'timeout').write_text(TIMEOUT); (work / 'bin' / 'timeout').chmod(0o755)
        (work / 'TASK.txt').write_text(text)
        procs.append(launch(run, i, work, text, model, backend=backend, deadline=meta['deadline']))
        meta['agents'].append(dict(id=i + 1, work=str(work)))
    (run / 'meta.json').write_text(json.dumps(meta, indent=2))
    with LOCK: PROCS[rid] = procs
    threading.Thread(target=supervise, args=(rid,), daemon=True).start()
    return rid


def records(run):
    f = run / 'records.jsonl'
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def canonical(points):
    return sorted(tuple(round(float(Fraction(str(c))), 6) for c in p) for p in points)


def history(recs, agent):
    mine = [r for r in recs if r['agent'] == agent]
    lines = [f'- attempt {j + 1} at {r["t"]:.0f}s: ' + (f'{r["score"]:.10f}' if r['valid'] else f'invalid ({r["error"]})')
             + (f'  [{r["dup"]}]' if r.get('dup') else '') for j, r in enumerate(mine)]
    best = max((r['score'] for r in mine if r['valid']), default=None)
    return '\n'.join(['Your attempts so far:'] + (lines or ['none']) + ([f'Best: {best:.10f}'] if best is not None else []))


def handle(run, meta, i, req):
    """Grade one attempt under the budget and append it to records.jsonl."""
    recs = records(run); inst = meta['instance']; agent = i + 1
    mine = [r for r in recs if r['agent'] == agent]
    t = round(time.time() - meta['started'], 1)
    if t > meta['minutes'] * 60: return dict(error='time is up')
    if len(mine) >= ATTEMPTS: return dict(error=f'no attempts left ({ATTEMPTS} used)')
    r = H.grade(inst, req['text'][:65536])
    rec = dict(agent=agent, kind='attempt', t=t, valid=r['valid'], score=r['score'], score_exact=r.get('score_exact'),
               error=r.get('error'), argmin=r.get('argmin', [])[:1], smallest=r.get('smallest'))
    if r['valid']:
        pts = json.loads(req['text']); rec['xy'] = [[float(Fraction(str(c))) for c in p] for p in pts]
        key = canonical(pts)
        same = next((j + 1 for j, o in enumerate(mine) if o['valid'] and canonical_xy(o['xy']) == key), None)
        if same: rec['dup'] = f'same layout as your attempt {same}'
    with (run / 'records.jsonl').open('a') as f: f.write(json.dumps(rec) + '\n')
    return dict(valid=r['valid'], score=r['score'], score_exact=r.get('score_exact'), error=r.get('error'),
                dup=rec.get('dup'), attempts_left=ATTEMPTS - len(mine) - 1, seconds_left=round(meta['minutes'] * 60 - t))


def canonical_xy(xy):
    return sorted(tuple(round(c, 6) for c in p) for p in xy)


def session_of(run, i):
    sid = None
    for line in (run / f'agent{i + 1}.jsonl').read_text(errors='replace').splitlines():
        if '"session_id"' in line:
            try: sid = json.loads(line).get('session_id') or sid
            except ValueError: pass
    return sid


def supervise(rid):
    """Serve grading requests, resume stopped agents, enforce the deadline."""
    run = RUNS / rid; meta = json.loads((run / 'meta.json').read_text())
    works = [Path(a['work']) for a in meta['agents']]; turns = [1] * AGENTS
    while True:
        with LOCK: procs = PROCS.get(rid, [])
        for i, w in enumerate(works):
            for req_file in sorted((w / 'outbox').glob('*.req')) if (w / 'outbox').exists() else []:
                try: req = json.loads(req_file.read_text())
                except ValueError: continue
                out = handle(run, meta, i, req) if req.get('kind') == 'attempt' else dict(error='bad request')
                (w / 'inbox' / (req_file.stem + '.json')).write_text(json.dumps(out, indent=1)); req_file.unlink()
        recs = records(run); now = time.time() - meta['started']
        stopped = (run / 'stopped').exists()
        if now > meta['minutes'] * 60 or stopped:
            for p in procs:
                if p: kill(p)
            if now > meta['minutes'] * 60: (run / 'deadline').write_text('1')
            if all(p is None or p.poll() is not None for p in procs): break
        else:
            for i, p in enumerate(procs):
                if p is None or p.poll() is None: continue
                attempts_left = ATTEMPTS - sum(r['agent'] == i + 1 for r in recs)
                sid = session_of(run, i)
                if attempts_left > 0 and turns[i] < MAX_TURNS and meta['minutes'] * 60 - now > 45 and sid:
                    turns[i] += 1
                    msg = (f'[Turn {turns[i]}] {meta["minutes"] * 60 - now:.0f} seconds left and {attempts_left} attempts '
                           f'left.\n\n{history(recs, i + 1)}\n\nKeep improving: try an idea different from what you '
                           'did so far, and spend attempts on candidates you expect to beat your best.')
                    with (run / f'agent{i + 1}.jsonl').open('a') as f:
                        f.write(json.dumps(dict(type='lab', subtype='resume', turn=turns[i], text=msg)) + '\n')
                    procs[i] = launch(run, i, works[i], msg, None if meta['model'] == 'CLI default' else meta['model'], sid,
                                      meta.get('backend', 'claude'), meta['deadline'])
                else: procs[i] = None  # done: no attempts, turns or time left
            if all(p is None or p.poll() is not None for p in procs): break
        with LOCK: PROCS[rid] = procs
        time.sleep(0.3)
    (run / 'turns.json').write_text(json.dumps(turns)); (run / 'ended').write_text(str(time.time()))


def kill(p):
    if p.poll() is None:
        try: os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError: pass


def stop(rid):
    (RUNS / rid / 'stopped').write_text('1')


def summarize(line):
    """Stream-json event -> short display items."""
    try: e = json.loads(line)
    except ValueError: return [dict(kind='raw', text=line[:500])]
    t = e.get('type')
    if t == 'lab': return [dict(kind='resume', text=f"— resumed (turn {e.get('turn')}) with its attempt history —")]
    if t == 'system' and e.get('subtype') == 'init': return [dict(kind='init', text=f"model {e.get('model')}")]
    items = []
    if t in ('assistant', 'user'):
        for c in e.get('message', {}).get('content', []) or []:
            if not isinstance(c, dict): continue
            if c.get('type') == 'text' and c.get('text', '').strip(): items.append(dict(kind='say', text=c['text'][:1500]))
            elif c.get('type') == 'thinking' and c.get('thinking', '').strip(): items.append(dict(kind='think', text='💭 ' + c['thinking'][-800:]))
            elif c.get('type') == 'tool_use':
                inp = c.get('input', {}); items.append(dict(kind='tool', text=f"{c.get('name')}: " + str(inp.get('command') or inp.get('file_path') or inp)[:600]))
            elif c.get('type') == 'tool_result':
                out = c.get('content'); out = out if isinstance(out, str) else ' '.join(x.get('text', '') for x in out or [] if isinstance(x, dict))
                items.append(dict(kind='out', text=out[:600]))
    elif t == 'result':
        items.append(dict(kind='result', text=f"{e.get('subtype')} · {str(e.get('result'))[:1500]}", cost=e.get('total_cost_usd')))
    return items


def state(rid):
    run = RUNS / rid; meta = json.loads((run / 'meta.json').read_text()); recs = records(run)
    with LOCK: procs = PROCS.get(rid, [])
    rewards = H.group_rewards([H.episode_terms([r for r in recs if r['agent'] == i + 1]) for i in range(len(meta['agents']))])
    agents = []
    for i, a in enumerate(meta['agents']):
        mine = [r for r in recs if r['agent'] == i + 1]
        best = lambda kind: {k: v for k, v in max((r for r in mine if r['kind'] == kind and r['valid']), key=lambda r: Fraction(r['score_exact']), default={}).items() if k != 'smallest'} or None
        log = run / f'agent{i + 1}.jsonl'
        events = [x for line in (log.read_text(errors='replace').splitlines() if log.exists() else []) for x in summarize(line)]
        alive = i < len(procs) and procs[i] is not None and procs[i].poll() is None
        agents.append(dict(id=i + 1, alive=alive, turns=1 + sum(e['kind'] == 'resume' for e in events),
                           attempts=len(mine), best_attempt=best('attempt'), reward=rewards[i],
                           records=[{k: v for k, v in r.items() if k not in ('xy', 'smallest')} for r in mine],
                           events=events[-100:], nevents=len(events)))
    return dict(meta={k: v for k, v in meta.items() if k != 'prompt'}, prompt=meta['prompt'], agents=agents,
                reward_weights=dict(final=H.W_FINAL, curve=H.W_CURVE, penalty=H.PENALTY, tie_cap=H.TIE_CAP),
                now=time.time(), ended=(run / 'ended').exists(), stopped=(run / 'stopped').exists(),
                deadline_hit=(run / 'deadline').exists())


def runs():
    return [json.loads((d / 'meta.json').read_text()) | dict(ended=(d / 'ended').exists())
            for d in sorted(RUNS.glob('*'), reverse=True) if (d / 'meta.json').exists()][:20]


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype='application/json'):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        path, _, q = self.path.partition('?'); params = dict(x.split('=', 1) for x in q.split('&') if '=' in x)
        try:
            if path == '/': self.send(200, (ROOT / 'hull_lab.html').read_text(), 'text/html; charset=utf-8')
            elif path == '/api/runs': self.send(200, [dict(id=r['id'], n=r['instance']['n'], k=r['instance']['k'],
                                                          seed=r['instance']['seed'], ended=r['ended']) for r in runs()])
            elif path == '/api/state' and params.get('run', '').replace('-', '').isalnum(): self.send(200, state(params['run']))
            elif path == '/api/preview':
                inst = (H.square_instance(int(params['k'])) if params.get('domain') == 'square'
                        else H.make_instance(int(params['n']), int(params['k']), int(params.get('seed', 0))))
                self.send(200, dict(instance=inst, prompt=task_text(inst, int(params.get('minutes', 5)))))
            else: self.send(404, dict(error='not found'))
        except (ValueError, KeyError, FileNotFoundError) as e: self.send(400, dict(error=str(e)))

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            if self.path == '/api/start':
                n, k = int(body['n']), int(body['k'])
                if not 2 <= n <= 4: raise ValueError('n must be 2..4')
                backend = body.get('agent', 'claude')
                minutes = max(1, min(30, int(body.get('minutes') or (15 if backend == 'qwen' else 5))))
                seed = None if body.get('seed') in (None, '') else int(body['seed'])
                model = str(body.get('model') or '').strip() or None
                if model and not model.replace('-', '').replace('.', '').isalnum(): raise ValueError('bad model name')
                if backend not in ('claude', 'qwen'): raise ValueError('agent must be claude or qwen')
                domain = body.get('domain', 'hull')
                if domain not in ('hull', 'square'): raise ValueError('domain must be hull or square')
                self.send(200, dict(run=start(n, k, seed, minutes, model, backend, domain)))
            elif self.path == '/api/stop': stop(body['run']); self.send(200, dict(ok=True))
            else: self.send(404, dict(error='not found'))
        except (ValueError, KeyError) as e: self.send(400, dict(error=str(e)))

    def log_message(self, *a): pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8771); a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    print(f'http://127.0.0.1:{a.port}', flush=True)
    server.serve_forever()
