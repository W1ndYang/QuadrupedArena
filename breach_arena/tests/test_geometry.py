import unittest

from breach_arena.geometry import boxes, clear_ray, line_of_sight, smoke_chord

LOW = boxes([dict(center=[0.0, 0.0], size=[0.4, 4.0, 0.7])])


class GeometryTest(unittest.TestCase):
    def test_box_blocks_segment(self):
        b = boxes([dict(center=[0, 0], size=[1, 1, 2.4])])
        self.assertFalse(clear_ray((-2, 0, 1), (2, 0, 1), b))
        self.assertTrue(clear_ray((-2, 2, 1), (2, 2, 1), b))
        self.assertTrue(clear_ray((-2, 0, 2.5), (2, 0, 2.5), b))

    def test_low_wall_hides_quad_close_behind_but_not_far(self):
        eye, quad = 1.05, 0.40
        self.assertFalse(line_of_sight((-4, 0), eye, (0.6, 0), quad, LOW))     # hugging the wall
        self.assertTrue(line_of_sight((-4, 0), eye, (8.0, 0), quad, LOW))      # far beyond it
        self.assertFalse(line_of_sight((-4, 0), quad, (8.0, 0), quad, LOW))    # quad-to-quad height stays blocked

    def test_smoke_cylinder(self):
        self.assertAlmostEqual(smoke_chord((-5, 0, 1), (5, 0, 1), (0, 0), 2.0, 2.5), 4.0, places=6)
        self.assertEqual(smoke_chord((-5, 3, 1), (5, 3, 1), (0, 0), 2.0, 2.5), 0.0)
        self.assertEqual(smoke_chord((-5, 0, 3), (5, 0, 3), (0, 0), 2.0, 2.5), 0.0)
        cloud = [dict(center=(0, 0), radius=2.0, height=2.5)]
        self.assertFalse(line_of_sight((-5, 0), 1.05, (5, 0), 0.4, [], cloud))


if __name__ == '__main__':
    unittest.main()
