"""Meaningful boundary, information-isolation and geometry checks for new rules."""
import copy
import math
import unittest

from fixed_goal_probe import (DEFAULT_RULES, TagClock, boxes, clear_ray, memory_update,
                              sampled_min_separation, smoke_blocks, smoke_chord, valid_lock)


class FixedGoalRulesTest(unittest.TestCase):
    def test_one_second_not_one_sample(self):
        c = TagClock()
        self.assertIsNone(c.step(.02, [True, False])[0])
        for _ in range(49):
            self.assertIsNone(c.step(.02, [True, False])[0])
        self.assertEqual(c.step(.02, [True, False])[:2], ('attacker', 'tag'))

    def test_break_destroys_accumulation(self):
        c = TagClock()
        for _ in range(45):
            c.step(.02, [True, False])
        c.step(.02, [False, False])
        for _ in range(20):
            self.assertIsNone(c.step(.02, [True, False])[0])

    def test_simultaneous_is_draw(self):
        c = TagClock(.04)
        c.step(.02, [True, True])
        c.step(.02, [True, True])
        self.assertEqual(c.step(.02, [True, True])[:2], ('draw', 'simultaneous'))

    def test_head_start_can_win(self):
        c = TagClock(.04)
        c.step(.02, [True, False])
        c.step(.02, [True, True])
        self.assertEqual(c.step(.02, [True, True])[:2], ('attacker', 'tag'))

    def test_goal_and_opposing_tag_same_tick_draw(self):
        c = TagClock(.02)
        c.step(.02, [False, True])
        self.assertEqual(c.step(.02, [False, True], goal=True)[:2], ('draw', 'simultaneous'))

    def test_goal_at_deadline_counts(self):
        self.assertEqual(TagClock().step(0, [False, False], goal=True, timeout=True)[:2], ('attacker', 'goal'))

    def test_timeout_defender(self):
        self.assertEqual(TagClock().step(0, [False, False], timeout=True)[:2], ('defender', 'timeout'))

    def test_shooter_speed_and_actual_tracking_error_bound_lock(self):
        cfg = DEFAULT_RULES
        self.assertFalse(valid_lock(True, 4, 0, 3.2, 0, True, cfg))
        # v0.3+: angular speed alone no longer breaks tracking. The actual
        # direction error must stay within five degrees, including while turning.
        self.assertTrue(valid_lock(True, 4, 0, 0, 1.2, True, cfg))
        self.assertFalse(valid_lock(True, 4, math.radians(5.1), 0, .1, True, cfg))
        self.assertTrue(valid_lock(True, 4, math.radians(4.9), 0, 1.2, True, cfg))
        self.assertTrue(valid_lock(True, 4, 0, .1, .1, True, cfg))

    def test_memory_is_stale_and_expires(self):
        seen = dict(position=[1., 2.], yaw=0.)
        m = memory_update(None, seen, 1, 4)
        seen['position'][0] = 99
        self.assertEqual(m['position'], [1., 2.])
        self.assertEqual(memory_update(m, None, 3, 4), m)
        self.assertIsNone(memory_update(m, None, 5.01, 4))

    def test_memory_never_authorizes_tag(self):
        self.assertFalse(valid_lock(False, 4, 0, 0, 0, True, DEFAULT_RULES))

    def test_smoke_symmetric_and_time_bounded(self):
        s = [dict(center=[0, 0], radius=2, height=2.5, active_at=1, expires_at=6)]
        a, b = [-4, 0, .4], [4, 0, 1.05]
        self.assertFalse(smoke_blocks(a, b, s, .99))
        self.assertTrue(smoke_blocks(a, b, s, 1))
        self.assertEqual(smoke_blocks(a, b, s, 2), smoke_blocks(b, a, s, 2))
        self.assertFalse(smoke_blocks(a, b, s, 6))

    def test_smoke_finite_height(self):
        self.assertEqual(smoke_chord([-3, 0, 4], [3, 0, 4], [0, 0], 2), 0)
        self.assertAlmostEqual(smoke_chord([-3, 0, 1], [3, 0, 1], [0, 0], 2), 4)

    def test_penetration_threshold_uses_path_length(self):
        s = [dict(center=[0, 0], radius=2, height=2.5, active_at=0, expires_at=5)]
        self.assertFalse(smoke_blocks([0, 0, 1], [.5, 0, 1], s, 1, 1.))
        self.assertTrue(smoke_blocks([-2, 0, 1], [2, 0, 1], s, 1, 1.))

    def test_3d_height_matters(self):
        wall = boxes([dict(center=[0, 0], size=[.4, 2, .7])])
        self.assertFalse(clear_ray([-1, 0, .4], [1, 0, .4], wall))
        self.assertTrue(clear_ray([-1, 0, 1.05], [1, 0, 1.05], wall))

    def test_swept_actor_collision_not_only_endpoints(self):
        self.assertAlmostEqual(sampled_min_separation([-1, 0], [1, 0], [1, 0], [-1, 0]), 0.)

    def test_small_target_can_be_body_blocked(self):
        separation = .7
        self.assertLess(.55, separation)
        self.assertGreater(DEFAULT_RULES['goal_radius_m'], separation)

    def test_frequency_consistent_tag_clock(self):
        for dt in (.01, .02, .05):
            c = TagClock()
            c.step(0, [True, False])
            result = None
            for _ in range(round(1/dt)):
                result = c.step(dt, [True, False])
            self.assertEqual(result[:2], ('attacker', 'tag'))


if __name__ == '__main__':
    unittest.main()
