import json
import random
import unittest
from fractions import Fraction

import hull_heilbronn as H


def diamond(k):
    """Hand-built n=2 instance with all axis points at distance 1: P = {|x|+|y| <= 1}, area 2."""
    one = ['1', '1']
    return dict(n=2, k=k, seed=-1, neg=one, pos=one, vertices=[], volume=str(H.volume(2, one, one)))


class HullHeilbronnTests(unittest.TestCase):
    def test_instance_deterministic_and_rational(self):
        a, b = H.make_instance(3, 7, 5), H.make_instance(3, 7, 5)
        self.assertEqual(a, b)
        self.assertNotEqual(a, H.make_instance(3, 7, 6))
        self.assertEqual(len(a['vertices']), 6)
        self.assertTrue(all(Fraction(1, 4) <= Fraction(r) <= 1 for r in a['neg'] + a['pos']))

    def test_volume_formula(self):
        self.assertEqual(Fraction(diamond(4)['volume']), 2)
        self.assertEqual(H.volume(3, ['1'] * 3, ['1'] * 3), Fraction(8, 6))  # octahedron

    def test_vertices_as_answer_score_matches_triangle(self):
        inst = diamond(4)
        r = H.grade(inst, json.dumps([[1, 0], [0, 1], [-1, 0], [0, -1]]))
        self.assertTrue(r['valid'])
        self.assertEqual(Fraction(r['score_exact']), Fraction(1, 2))  # each triangle area 1, / area 2

    def test_membership_exact(self):
        inst = diamond(4)
        self.assertTrue(H.grade(inst, json.dumps([['1/2', '1/2'], [0, 1], [-1, 0], [0, -1]]))['valid'])
        bad = H.grade(inst, json.dumps([['1/2', '0.5000000000001'], [0, 1], [-1, 0], [0, -1]]))
        self.assertFalse(bad['valid']); self.assertEqual(bad['score'], 0.0)

    def test_collinear_scores_zero(self):
        r = H.grade(diamond(4), json.dumps([[0, 0], ['1/2', 0], ['-1/2', 0], [0, 1]]))
        self.assertEqual(Fraction(r['score_exact']), 0)

    def test_invalid_shapes(self):
        inst = diamond(4)
        for text in ['[[0,0]]', '[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]', 'not json', '[[true,0],[0,0],[0,0],[0,0]]']:
            self.assertFalse(H.grade(inst, text)['valid'])

    def test_three_dims_tetra_volume(self):
        inst = dict(n=3, k=4, seed=-1, neg=['1'] * 3, pos=['1'] * 3, vertices=[], volume=str(H.volume(3, ['1'] * 3, ['1'] * 3)))
        r = H.grade(inst, json.dumps([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.assertEqual(Fraction(r['min_volume']), Fraction(1, 6))
        self.assertEqual(Fraction(r['score_exact']), Fraction(1, 8))

    def test_random_points_inside_and_baseline_positive(self):
        inst = H.make_instance(2, 6, 1)
        pts = H.random_points(inst, random.Random(0))
        self.assertTrue(all(H.inside(inst, p) for p in pts))
        self.assertGreater(H.baseline(inst, 20), 0)

    def test_env_roundtrip_extracts_answer_from_prose(self):
        env = H.HullHeilbronnEnv(k_range=(4, 4), baseline_samples=10, seed=3)
        text = env.reset()
        self.assertIn('JSON list of 4 points', text)
        reward, done, info = env.step('Here you go: [[0,0],[0,0],[0,0],[0,0]] final')
        self.assertTrue(done); self.assertTrue(info['valid']); self.assertEqual(reward, 0.0)
        self.assertIn('ratio', info)

    def test_tiebreak_never_overturns_larger_min(self):
        a = H.episode_terms([dict(valid=True, score_exact='0.0500', smallest=[0.05] * 30)])  # many ties
        b = H.episode_terms([dict(valid=True, score_exact='0.0499', smallest=[0.0499] + [0.09] * 29)])  # 0.2% lower, spread
        self.assertGreater(a['final'], b['final'])
        self.assertLessEqual(b['bonus'], H.TIE_CAP * b['best'] + 1e-15)

    def test_tiebreak_orders_equal_minimums(self):
        tight = H.episode_terms([dict(valid=True, score_exact='0.05', smallest=[0.05] * 10)])
        spread = H.episode_terms([dict(valid=True, score_exact='0.05', smallest=[0.05] + [0.051] * 9)])
        self.assertGreater(spread['final'], tight['final'])

    def test_curve_rewards_early_progress_and_penalties(self):
        early = H.episode_terms([dict(valid=True, score_exact='0.05', smallest=[])])
        late = H.episode_terms([dict(valid=True, score_exact='0.01', smallest=[])] * 9 + [dict(valid=True, score_exact='0.05', smallest=[])])
        self.assertEqual(early['best'], late['best']); self.assertGreater(early['curve'], late['curve'])
        wasted = H.episode_terms([dict(valid=False), dict(valid=True, score_exact='0.05', smallest=[], dup='x')])
        self.assertEqual((wasted['invalid'], wasted['dup']), (1, 1))

    def test_group_rewards_scale_free_and_near_ties(self):
        mk = lambda v: H.episode_terms([dict(valid=True, score_exact=v, smallest=[])])
        small = H.group_rewards([mk('0.0003'), mk('0.0002'), mk('0.0001')])
        big = H.group_rewards([mk('0.3'), mk('0.2'), mk('0.1')])
        for x, y in zip(small, big): self.assertAlmostEqual(x['advantage'], y['advantage'], places=6)
        tie = H.group_rewards([mk('0.07078470000'), mk('0.07078470001'), mk('0.07078469999')])
        self.assertTrue(all(t['advantage'] == 0.0 for t in tie))
        self.assertEqual(H.group_rewards([H.episode_terms([]), mk('0.1')])[0]['best'], 0.0)

    def test_not_submitting_never_beats_invalid(self):
        none = H.episode_terms([])
        one_bad = H.episode_terms([dict(valid=False)])
        five_bad = H.episode_terms([dict(valid=False)] * 5)
        good = H.episode_terms([dict(valid=True, score_exact='0.05', smallest=[])])
        r = H.group_rewards([none, one_bad, five_bad, good])
        self.assertLessEqual(r[0]['reward'], r[1]['reward'])
        self.assertLessEqual(r[0]['reward'], r[2]['reward'])
        self.assertGreater(r[3]['reward'], r[1]['reward'])
        zero = H.episode_terms([dict(valid=True, score_exact='0', smallest=[])])   # valid but degenerate: no floor
        self.assertGreater(H.group_rewards([zero, good])[0]['reward'], r[0]['reward'] - 1)


if __name__ == '__main__':
    unittest.main()
