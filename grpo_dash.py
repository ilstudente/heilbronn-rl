"""Live dashboard for hull_grpo.py runs (Modal volume `hull-grpo`).

python3 grpo_dash.py [--port 8775]     then open http://127.0.0.1:8775

A background thread copies runs/<name>/{config.json,metrics.jsonl,rollouts/*.json} from the volume into
grpo_runs/<name>/ every 30 s (read-only `modal volume get`; no GPU, no writes to Modal).
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / 'grpo_runs'
MODAL = str(ROOT.parent / '.venv' / 'bin' / 'modal')
ENV = {**os.environ, 'NO_COLOR': '1', 'TERM': 'dumb'}
RUN_RE = re.compile(r'runs/(\d{8}-\d{6}[\w-]*)')
LOCK = threading.Lock()
STATUS = {'last_sync': None, 'error': None}


def modal(*args, timeout=120):
    return subprocess.run([MODAL, *args], capture_output=True, text=True, env=ENV, timeout=timeout, stdin=subprocess.DEVNULL)


def remote_runs():
    return sorted(set(RUN_RE.findall(modal('volume', 'ls', 'hull-grpo', 'runs').stdout)))


def sync_run(name, full=False):
    d = LOCAL / name
    (d / 'rollouts').mkdir(parents=True, exist_ok=True)
    for f in ('config.json', 'metrics.jsonl'):
        modal('volume', 'get', '--force', 'hull-grpo', f'runs/{name}/{f}', str(d / f))
    listed = set(re.findall(r'rollouts/([\w.-]+\.json)', modal('volume', 'ls', 'hull-grpo', f'runs/{name}/rollouts').stdout))
    for f in sorted(listed):
        if full or not (d / 'rollouts' / f).exists():
            modal('volume', 'get', '--force', 'hull-grpo', f'runs/{name}/rollouts/{f}', str(d / 'rollouts' / f))


def syncer():
    first = True
    while True:
        try:
            runs = remote_runs()
            for i, name in enumerate(runs):
                if first or i == len(runs) - 1 or not (LOCAL / name / 'metrics.jsonl').exists():
                    sync_run(name, full=first)
            first = False
            STATUS.update(last_sync=time.time(), error=None)
        except Exception as e:                                   # noqa: BLE001 - keep serving what we have
            STATUS['error'] = f'{type(e).__name__}: {e}'
        time.sleep(30)


def run_state(name):
    d = LOCAL / name
    cfg = json.loads((d / 'config.json').read_text()) if (d / 'config.json').exists() else {}
    metrics = [json.loads(l) for l in (d / 'metrics.jsonl').read_text().splitlines() if l.strip()] \
        if (d / 'metrics.jsonl').exists() else []
    rollouts = {}
    for f in sorted((d / 'rollouts').glob('*.json')) if (d / 'rollouts').exists() else []:
        try: rollouts[f.stem] = json.loads(f.read_text())
        except ValueError: pass
    return dict(name=name, config=cfg, metrics=metrics, rollouts=rollouts, status=STATUS)


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype='application/json'):
        data = body.encode() if isinstance(body, str) else json.dumps(body, default=str).encode()
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        path, _, q = self.path.partition('?'); params = dict(x.split('=', 1) for x in q.split('&') if '=' in x)
        if path == '/': return self.send(200, (ROOT / 'grpo_dash.html').read_text(), 'text/html; charset=utf-8')
        if path == '/api/runs':
            return self.send(200, dict(runs=sorted((p.name for p in LOCAL.glob('*') if p.is_dir()), reverse=True), status=STATUS))
        if path == '/api/run' and RUN_RE.match('runs/' + params.get('name', '')):
            return self.send(200, run_state(params['name']))
        self.send(404, dict(error='not found'))

    def log_message(self, *a): pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8775); a = ap.parse_args()
    LOCAL.mkdir(exist_ok=True)
    threading.Thread(target=syncer, daemon=True).start()
    server = ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    print(f'http://127.0.0.1:{a.port}', flush=True)
    server.serve_forever()
