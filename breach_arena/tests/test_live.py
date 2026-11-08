import os
import tempfile
import unittest

try:
    import pygame  # noqa: F401
    HAVE_PYGAME = True
except ImportError:
    HAVE_PYGAME = False


@unittest.skipUnless(HAVE_PYGAME, 'pygame not installed (optional: breach_arena/requirements-live.txt)')
class LiveClient(unittest.TestCase):
    def test_headless_player_with_aim_assist(self):
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        from breach_arena.live import play_live

        def player(t, p):
            obs = p.last_obs
            if obs and obs['opponent']:
                q = obs['opponent']['xy']
                p.state.update(move=[0.0, 0.0], aim=[q[0] + 0.5, q[1] + 0.5], fire=True)   # rough aim
            else:
                p.state.update(move=[0.0, 0.0], aim=[0.0, 0.0], fire=False)
        with tempfile.TemporaryDirectory() as d:
            res = play_live('BA-1', 'rule', rounds=1, out_dir=d, headless_seconds=60, script_input=player)
            self.assertEqual(len(res), 1)
            self.assertIn(res[0]['result']['winner'], ('human', 'quadruped'))
            self.assertTrue(os.path.exists(os.path.join(d, 'round1.viewer.html')))


if __name__ == '__main__':
    unittest.main()
