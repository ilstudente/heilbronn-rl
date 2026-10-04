"""Minimal tool-calling agent for the Hull Heilbronn Lab, talking to an OpenAI-compatible endpoint (the Qwen server
in qwen_serve.py). Drop-in for `claude -p` in hull_lab.py: the prompt arrives on stdin, events go to stdout in the
same stream-json shapes, and `--resume SESSION` continues a saved conversation.

    echo "task" | python3 qwen_agent.py --base-url https://…modal.run/v1 [--model qwen] [--resume ID]

The API key comes from QWEN_API_KEY. Tools: bash (runs in the working directory, time-limited) and write_file.
Conversations are saved in .sessions/<id>.json in the working directory.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

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
SYSTEM = ('You are an autonomous agent working in a sandboxed directory. Use the tools to run code and inspect '
          'results. Keep going until the task is done, then give a short final answer without a tool call.')
emit = lambda e: print(json.dumps(e), flush=True)


def chat(args, messages):
    body = dict(model=args.model, messages=messages, tools=TOOLS, tool_choice='auto', max_tokens=args.max_tokens,
                temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=args.reasoning_effort)
    req = urllib.request.Request(args.base_url.rstrip('/') + '/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json',
                                  'Authorization': f"Bearer {os.environ.get('QWEN_API_KEY', '')}"})
    with urllib.request.urlopen(req, timeout=args.request_timeout) as r:
        return json.load(r)


def run_tool(name, a):
    if name == 'write_file':
        path = Path(a['path'])
        if path.is_absolute() or '..' in path.parts: return 'error: path must stay inside the working directory'
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(a['content'])
        return f'wrote {len(a["content"])} characters to {path}'
    if name == 'bash':
        limit = max(1, min(150, int(a.get('timeout') or 120)))
        try:
            p = subprocess.run(['zsh', '-c', a['command']], capture_output=True, text=True, timeout=limit)
            out = (p.stdout + p.stderr).strip() or '(no output)'
            return (out if len(out) <= 8000 else out[:3000] + '\n…[truncated]…\n' + out[-4000:]) + f'\n[exit {p.returncode}]'
        except subprocess.TimeoutExpired:
            return f'error: command timed out after {limit}s'
    return f'error: unknown tool {name}'


def footer():
    """Footer on every tool result: time left (HULL_DEADLINE), attempts left and best score (from inbox/ answers)."""
    import time
    parts = []
    if os.environ.get('HULL_DEADLINE'):
        left = max(0, int(float(os.environ['HULL_DEADLINE']) - time.time()))
        parts.append(f'{left // 60}m{left % 60:02d}s left')
    answers = []
    for f in Path('inbox').glob('*.json') if Path('inbox').is_dir() else []:
        try: answers.append(json.loads(f.read_text()))
        except ValueError: pass
    if answers or os.environ.get('HULL_DEADLINE'):
        used = sum('attempts_left' in a for a in answers)
        left_a = min((a['attempts_left'] for a in answers if 'attempts_left' in a), default=None)
        best = max((a['score'] for a in answers if a.get('valid')), default=None)
        parts.append(f'{left_a} attempts left' if left_a is not None else f'{used} attempts used')
        parts.append(f'your best: {best:.6f}' if best is not None else 'your best: none yet, submit an attempt')
    return f"\n[{' · '.join(parts)}]" if parts else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', required=True); ap.add_argument('--model', default='qwen')
    ap.add_argument('--resume'); ap.add_argument('--max-steps', type=int, default=60)
    ap.add_argument('--max-tokens', type=int, default=16384); ap.add_argument('--request-timeout', type=int, default=900)
    ap.add_argument('--reasoning-effort', default=os.environ.get('QWEN_REASONING_EFFORT', 'medium'))
    args = ap.parse_args()
    sessions = Path('.sessions'); sessions.mkdir(exist_ok=True)
    sid = args.resume or uuid.uuid4().hex
    store = sessions / f'{sid}.json'
    messages = json.loads(store.read_text()) if args.resume and store.exists() else [dict(role='system', content=SYSTEM)]
    messages.append(dict(role='user', content=sys.stdin.read()))
    emit(dict(type='system', subtype='init', model=f'{args.model} @ {args.base_url}', session_id=sid))
    final, status = '', 'success'
    for _ in range(args.max_steps):
        try:
            msg = chat(args, messages)['choices'][0]['message']
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            final, status = f'model request failed: {e}', 'error'; break
        thinking = msg.get('reasoning_content') or msg.get('reasoning') or ''
        calls = msg.get('tool_calls') or []
        content = [dict(type='thinking', thinking=thinking)] if thinking else []
        if msg.get('content'): content.append(dict(type='text', text=msg['content']))
        content += [dict(type='tool_use', name=c['function']['name'], input=_args(c)) for c in calls]
        emit(dict(type='assistant', message=dict(content=content)))
        messages.append({k: v for k, v in dict(role='assistant', content=msg.get('content') or '', tool_calls=calls or None,
                                               reasoning_content=thinking or None).items() if v is not None})
        if not calls:
            final = msg.get('content') or ''; break
        for c in calls:
            out = run_tool(c['function']['name'], _args(c)) + footer()
            emit(dict(type='user', message=dict(content=[dict(type='tool_result', content=out)])))
            messages.append(dict(role='tool', tool_call_id=c.get('id', ''), content=out))
        store.write_text(json.dumps(messages))
    else:
        final, status = f'stopped after {args.max_steps} steps', 'error_max_turns'
    store.write_text(json.dumps(messages))
    emit(dict(type='result', subtype=status, result=final, session_id=sid))


def _args(call):
    try: return json.loads(call['function'].get('arguments') or '{}')
    except ValueError: return {}


if __name__ == '__main__':
    main()
