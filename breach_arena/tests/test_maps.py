import unittest

from breach_arena import MAP_SCHEMA
from breach_arena.config import GameMap, available_maps, load_rules
from breach_arena.geometry import boxes, line_of_sight
from breach_arena.humans import make_bot
from breach_arena.nav import Planner


class Maps(unittest.TestCase):
    def test_maps_are_consistent(self):
        r = load_rules()
        self.assertEqual(available_maps(), ['BA-1', 'BA-2'])
        for mid in available_maps():
            with self.subTest(map=mid):
                m = GameMap.load(mid)
                self.assertEqual(m.data['schema'], MAP_SCHEMA)
                nav = Planner(m.covers, m.size, r.planner.clearance_m, r.planner.grid_m)
                for k, p in m.points.items():
                    self.assertTrue(nav.point_ok(p), f'{k} not walkable')
                q, h = m.spawns['quadruped']['xy'], m.spawns['human']['xy']
                self.assertTrue(nav.point_ok(q) and nav.point_ok(h))
                self.assertFalse(line_of_sight(h, 1.05, q, 0.4, boxes(m.covers)), 'spawns see each other')
                self.assertIsNotNone(nav.plan(q, m.goal))
                plays = m.plays
                for t, T in plays['throws'].items():
                    self.assertEqual(m.kinds[t], 'throw')
                    for v in T['run_via'] + plays['approaches'].get(t, []):
                        m.resolve(v)
                for name, L in plays['lures'].items():
                    for v in [L['feint'], L['staging']] + L['relocate_via']:
                        m.resolve(v)
                for c in m.data['cycle_checks']:
                    make_bot(c['human'])
                    self.assertIn(c['expect'], ('quadruped', 'human'))


if __name__ == '__main__':
    unittest.main()
