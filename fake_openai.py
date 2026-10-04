"""Fake OpenAI-compatible chat server for testing qwen_agent.py / hull_lab.py without a GPU.

python3 fake_openai.py [--port 8790]    (expects Authorization: Bearer test-key)
Each conversation: the model 'thinks', calls bash to write a random in-hull candidate and grade it with
attempt.py, then answers without a tool call.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = r'''python3 - <<'EOF'
import json, random
from fractions import Fraction as F
i = json.load(open('instance.json')); neg = [F(a) for a in i['neg']]; pos = [F(b) for b in i['pos']]
pts = []
while len(pts) < i['k']:
    x = [F(random.randint(int(-a * 1000), int(b * 1000)), 1000) for a, b in zip(neg, pos)]
    if sum(abs(c) / (pos[j] if c >= 0 else neg[j]) for j, c in enumerate(x)) <= 1: pts.append([str(c) for c in x])
json.dump(pts, open('cand.json', 'w'))
EOF
python3 attempt.py cand.json'''


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.headers.get('Authorization') != 'Bearer test-key':
            return self.reply(401, dict(error='bad key'))
        tools_used = sum(m['role'] == 'tool' for m in body['messages'][-4:])
        if body['messages'][-1]['role'] == 'user' or not tools_used:
            msg = dict(role='assistant', content='', reasoning_content='I will try a random layout first.',
                       tool_calls=[dict(id='call1', type='function', function=dict(name='bash', arguments=json.dumps(dict(command=SCRIPT))))])
        else:
            msg = dict(role='assistant', content='Random layout submitted; done for this turn.', reasoning_content='Graded.')
        self.reply(200, dict(choices=[dict(index=0, message=msg, finish_reason='stop')]))

    def reply(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def log_message(self, *a): pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8790); a = ap.parse_args()
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
