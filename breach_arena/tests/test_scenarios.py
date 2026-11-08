"""The designed play / counter-play loop, run through the real referee (deterministic)."""
import unittest

from breach_arena.config import available_maps, GameMap
from breach_arena.runner import run_matrix


class CycleMatrix(unittest.TestCase):
    def test_every_map_matches_its_cycle_checks(self):
        for mid in available_maps():
            with self.subTest(map=mid):
                rows = run_matrix(mid)
                self.assertEqual(len(rows), len(GameMap.load(mid).data['cycle_checks']))
                bad = [(r['id'], r['expect'], r['winner'], r['reason']) for r in rows if not r['ok']]
                self.assertEqual(bad, [])

    def test_ba1_covers_the_four_way_cycle(self):
        ids = {c['id'] for c in GameMap.load('BA-1').data['cycle_checks']}
        for needed in ('rule_vs_hold', 'rule_vs_hunt', 'flank_without_smoke_vs_hold', 'smoke_on_hunter_side_vs_hunt',
                       'goal_smoke_vs_standoff_NE', 'path_smoke_vs_standoff_NE'):
            self.assertIn(needed, ids)


if __name__ == '__main__':
    unittest.main()
