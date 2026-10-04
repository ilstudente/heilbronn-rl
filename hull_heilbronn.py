"""Heilbronn problem in random axis-hull polytopes: instance generator, exact grader and RL environment.

Instance (n, k, seed): on each of the n coordinate axes pick one random rational point on the negative side
(-a_i) and one on the positive side (+b_i), with a_i, b_i in [lo, 1]. The domain P is the convex hull of these
2n points. Because each axis has one point on each side of the origin, P = {x : sum_i |x_i| / r_i(x_i) <= 1},
where r_i(x_i) = b_i if x_i >= 0 else a_i. That gives an exact membership test with no hull library.

Task: place k points in P maximising the minimum volume over all (n+1)-point simplices (triangle area when n=2).
Score = min simplex volume / vol(P), exact rational, invariant to the instance's scale. vol(P) = prod(a_i+b_i)/n!.
Invalid answers (wrong shape, outside P, bad number format) score 0 with an error.

CLI:
  python hull_heilbronn.py make  --n 2 --k 8 --seed 0 [--out inst.json]   # instance JSON + prompt
  python hull_heilbronn.py grade --instance inst.json --file cand.json     # exact grade
  python hull_heilbronn.py baseline --n 2 --k 8 --seed 0 [--samples 200]   # best-of-random reference score
"""
import argparse
import json
import math
import random
import re
from decimal import Decimal
from fractions import Fraction
from itertools import combinations

MAX_SUBSETS = 300_000


def make_instance(n, k, seed, lo=Fraction(1, 4), den=64):
    """Random axis hull. Coordinates are rationals with denominator `den`, drawn from [lo, 1]."""
    if n < 2: raise ValueError('n must be >= 2')
    if k < n + 1: raise ValueError('k must be >= n+1')
    if math.comb(k, n + 1) > MAX_SUBSETS: raise ValueError(f'C(k,n+1) exceeds {MAX_SUBSETS}')
    rng = random.Random(f'{n}-{k}-{seed}')
    draw = lambda: Fraction(rng.randint(math.ceil(lo * den), den), den)
    neg, pos = [draw() for _ in range(n)], [draw() for _ in range(n)]
    vertices = []
    for i in range(n):
        for r in (-neg[i], pos[i]):
            v = [Fraction(0)] * n; v[i] = r; vertices.append(v)
    return dict(n=n, k=k, seed=seed, neg=[str(a) for a in neg], pos=[str(b) for b in pos],
                vertices=[[str(c) for c in v] for v in vertices], volume=str(volume(n, neg, pos)))


SQUARE_RECORDS = {16: ('7/341', 'https://arxiv.org/html/2603.11107v1#A1 (Appendix A, Table 13)')}


def square_instance(k=16):
    """The classic Heilbronn problem: k points in the unit square [0,1]^2 (area 1, so score = min triangle area)."""
    if math.comb(k, 3) > MAX_SUBSETS: raise ValueError(f'C(k,3) exceeds {MAX_SUBSETS}')
    rec = SQUARE_RECORDS.get(k)
    return dict(kind='square', n=2, k=k, seed='square', vertices=[['0', '0'], ['1', '0'], ['1', '1'], ['0', '1']],
                volume='1', **(dict(record=rec[0], record_source=rec[1]) if rec else {}))


def volume(n, neg, pos):
    return math.prod(Fraction(a) + Fraction(b) for a, b in zip(neg, pos)) / math.factorial(n)


def inside(inst, x):
    if inst.get('kind') == 'square': return all(0 <= c <= 1 for c in x)
    neg, pos = [Fraction(a) for a in inst['neg']], [Fraction(b) for b in inst['pos']]
    return sum(abs(c) / (pos[i] if c >= 0 else neg[i]) for i, c in enumerate(x)) <= 1


def det(m):
    """Exact determinant by Fraction Gaussian elimination."""
    m = [row[:] for row in m]; n = len(m); d = Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if m[r][c] != 0), None)
        if p is None: return Fraction(0)
        if p != c: m[c], m[p] = m[p], m[c]; d = -d
        d *= m[c][c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            if f: m[r] = [a - f * b for a, b in zip(m[r], m[c])]
    return d


def simplex_volume(pts):
    base = pts[0]
    return abs(det([[a - b for a, b in zip(p, base)] for p in pts[1:]])) / math.factorial(len(base))


def coordinate(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError('Coordinates must be decimal numbers or rational strings')
    s = str(value)
    if len(s) > 100 or not re.fullmatch(r'[+-]?(?:\d+/\d+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,3})?)', s):
        raise ValueError('Invalid or oversized coordinate')
    try: return Fraction(s)
    except (ValueError, ZeroDivisionError, OverflowError): raise ValueError('Invalid coordinate') from None


def evaluate(inst, points):
    n, k = inst['n'], inst['k']
    if not isinstance(points, list) or len(points) != k: raise ValueError(f'Expected exactly {k} points')
    if any(not isinstance(p, list) or len(p) != n for p in points): raise ValueError(f'Each point must have {n} coordinates')
    p = [[coordinate(v) for v in row] for row in points]
    outside = [i for i, x in enumerate(p) if not inside(inst, x)]
    if outside: raise ValueError(f'Points outside the hull: {outside}')
    vols = [(simplex_volume([p[i] for i in s]), list(s)) for s in combinations(range(k), n + 1)]
    low = min(v for v, _ in vols)
    vol = Fraction(inst['volume']); score = low / vol
    smallest = sorted(v for v, _ in vols)[:2 * k]  # for the reward's tie-break
    return dict(valid=True, score_exact=str(score), score=float(score), min_volume=str(low),
                argmin=[s for v, s in vols if v == low], subsets=len(vols), smallest=[float(v / vol) for v in smallest])


def grade(inst, text):
    """Grade a JSON list of k points. Never raises; invalid -> valid=False, score 0."""
    try:
        pts = json.loads(text, parse_float=Decimal,
                         parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite number')))
        return evaluate(inst, pts)
    except (ValueError, TypeError) as e:
        return dict(valid=False, score=0.0, error=str(e))


def prompt(inst):
    if inst.get('kind') == 'square':
        rec = f" The best known value for k={inst['k']} is {inst['record']}." if inst.get('record') else ''
        return (f"Classic Heilbronn problem: place {inst['k']} points in the unit square [0,1]^2 (boundary allowed) to "
                f"maximise the smallest area of any triangle formed by three of them. Score = that minimum area, graded "
                f"exactly.{rec}\nAnswer with only a JSON list of {inst['k']} points, each a list of 2 numbers or "
                f"fraction strings like \"3/7\".")
    n, k = inst['n'], inst['k']
    what = 'triangle' if n == 2 else f'{n}-simplex ({n + 1} points)'
    return (f"Let P be the convex hull of these {2 * n} points in R^{n}:\n"
            + '\n'.join('  (' + ', '.join(v) + ')' for v in inst['vertices'])
            + f"\nEquivalently, x is in P iff sum_i |x_i| / r_i <= 1 with r_i = {inst['pos']}[i] if x_i >= 0 "
              f"else {inst['neg']}[i].\n"
            f"Place {k} points in P (boundary allowed) to maximise the smallest area/volume of any {what} "
            f"formed by them. Score = that minimum / vol(P), with vol(P) = {inst['volume']}, graded exactly.\n"
            f"Answer with only a JSON list of {k} points, each a list of {n} numbers or fraction strings like \"3/7\".")


def extract_answer(text):
    """Last top-level JSON list of lists in a model response."""
    for m in reversed(list(re.finditer(r'\[\s*\[', text))):
        depth = 0
        for j in range(m.start(), len(text)):
            depth += text[j] == '['; depth -= text[j] == ']'
            if depth == 0: return text[m.start():j + 1]
    return text


def random_points(inst, rng, den=10**6):
    """Uniform rational points in P by rejection from the bounding box."""
    if inst.get('kind') == 'square':
        return [[Fraction(rng.randint(0, den), den) for _ in range(2)] for _ in range(inst['k'])]
    neg, pos = [Fraction(a) for a in inst['neg']], [Fraction(b) for b in inst['pos']]
    out = []
    while len(out) < inst['k']:
        x = [Fraction(rng.randint(int(-a * den), int(b * den)), den) for a, b in zip(neg, pos)]
        if inside(inst, x): out.append(x)
    return out


def baseline(inst, samples=200, seed=0):
    rng = random.Random(seed)
    return max(Fraction(evaluate(inst, [[str(c) for c in p] for p in random_points(inst, rng)])['score_exact'])
               for _ in range(samples))


# Reward for a group of agents that each spend up to BUDGET exact attempts on the same instance.
BUDGET, W_FINAL, W_CURVE, PENALTY, TIE_CAP, TIE_REL, STD_FLOOR = 10, 0.8, 0.2, 0.05, 1e-3, 1e-6, 1e-3
NO_VALID = 0.25   # penalty floor for an episode without a valid attempt: never submitting is never better than trying


def episode_terms(attempts, budget=BUDGET):
    """One agent's attempts (in order: dicts with valid, score_exact, smallest, dup) -> reward terms.

    best   exact best min-score over valid attempts (the objective).
    bonus  leximin tie-break: mean gap of the next-smallest normalised volumes above the minimum, capped at
           TIE_CAP * best. It only reorders layouts whose minimums are within 0.1%; a larger minimum always wins.
    final  best + bonus.
    curve  mean of the best-so-far over the budget's attempt slots (unused slots count at the final best),
           rewarding reaching good layouts early.
    invalid / dup  wasted attempts: invalid submissions and repeats of the agent's own earlier layout."""
    running, curve, best, top = 0.0, [], None, None
    for a in attempts[:budget]:
        if a.get('valid') and (best is None or Fraction(a['score_exact']) > best):
            best, top = Fraction(a['score_exact']), a
        running = float(best) if best is not None else 0.0
        curve.append(running)
    curve += [running] * (budget - len(curve))
    b = float(best) if best is not None else 0.0
    rest = (top or {}).get('smallest') or []
    gap = sum(v - b for v in rest[1:]) / (len(rest) - 1) if len(rest) > 1 else 0.0
    bonus = TIE_CAP * b * min(1.0, gap / b) if b > 0 else 0.0
    return dict(best=b, bonus=bonus, final=b + bonus, curve=sum(curve) / budget, attempts=len(attempts),
                valid=sum(bool(a.get('valid')) for a in attempts),
                invalid=sum(not a.get('valid') for a in attempts), dup=sum(bool(a.get('dup')) for a in attempts))


def group_rewards(terms):
    """Group-relative rewards and GRPO-style advantages for agents on the same instance.

    r = W_FINAL * final/M + W_CURVE * curve/M - penalty, penalty = PENALTY * (invalid + dup), or at least NO_VALID
    when there is no valid attempt (so skipping attempts can't dodge the invalid penalty). M is the group's best final, so the
    reward is scale-free per instance and needs no external reference. Ratios are rounded to 1e-6 so near-ties
    tie, and the advantage divides by max(std, STD_FLOOR) so rounding noise isn't blown up."""
    m = max((t['final'] for t in terms), default=0.0)
    ratio = lambda v: round(v / m, 6) if m > 0 else 0.0
    pen = lambda t: max(PENALTY * (t['invalid'] + t['dup']), NO_VALID if t['best'] <= 0 and t.get('valid', 1) == 0 else 0.0)
    r = [W_FINAL * ratio(t['final']) + W_CURVE * ratio(t['curve']) - pen(t) for t in terms]
    mean = sum(r) / len(r) if r else 0.0
    std = (sum((x - mean) ** 2 for x in r) / len(r)) ** 0.5 if r else 0.0
    return [dict(t, reward=x, advantage=0.0 if abs(x - mean) < TIE_REL else (x - mean) / max(std, STD_FLOOR))
            for t, x in zip(terms, r)]


class HullHeilbronnEnv:
    """Single-step environment: reset() -> prompt, step(response) -> reward in [0, 1).

    reward = exact normalised score; if baseline_samples > 0, info also carries the best-of-random reference and
    the ratio to it, which is a steadier learning signal across instances than the raw score."""

    def __init__(self, n_range=(2, 2), k_range=(5, 10), baseline_samples=0, seed=0):
        self.n_range, self.k_range, self.baseline_samples = n_range, k_range, baseline_samples
        self.rng = random.Random(seed); self.inst = None

    def reset(self, n=None, k=None, seed=None):
        n = n or self.rng.randint(*self.n_range); k = k or self.rng.randint(*self.k_range)
        seed = self.rng.randrange(2**31) if seed is None else seed
        self.inst = make_instance(n, k, seed)
        self.ref = baseline(self.inst, self.baseline_samples) if self.baseline_samples else None
        return prompt(self.inst)

    def step(self, response):
        result = grade(self.inst, extract_answer(response))
        info = dict(result, instance=self.inst)
        if self.ref: info.update(reference=float(self.ref), ratio=result['score'] / float(self.ref))
        return result['score'], True, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['make', 'grade', 'baseline'])
    ap.add_argument('--n', type=int); ap.add_argument('--k', type=int); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out'); ap.add_argument('--instance'); ap.add_argument('--file')
    ap.add_argument('--samples', type=int, default=200)
    a = ap.parse_args()
    if a.action == 'grade':
        inst = json.load(open(a.instance)); print(json.dumps(grade(inst, extract_answer(open(a.file).read())), indent=2)); return
    if a.n is None or a.k is None: ap.error('--n and --k required')
    inst = make_instance(a.n, a.k, a.seed)
    if a.action == 'make':
        if a.out: json.dump(inst, open(a.out, 'w'), indent=2)
        print(json.dumps(inst, indent=2)); print(); print(prompt(inst))
    else:
        b = baseline(inst, a.samples); print(f'best of {a.samples} random: {b} ~ {float(b):.6g}')


if __name__ == '__main__':
    main()
