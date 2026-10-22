"""CPU-only checks for the presentation's scripted accounting and geometry."""
import json
from pathlib import Path
import unittest
from demo_timeline import build, visible, segment_box, pose
import numpy as np


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.config = json.loads((root/"configs/demo_v1.json").read_text())
        cls.arena = json.loads((root/"configs/arena_v0.json").read_text())
        cls.states, cls.events, cls.validation = build(cls.config, cls.arena)

    def test_occlusion_is_three_dimensional(self):
        cover = [{"center":[0,0],"size":[.4,2.,.6]}]
        self.assertFalse(visible([-1,0,.3],[1,0,.3],cover))
        self.assertTrue(visible([-1,0,1.2],[1,0,1.2],cover))
        self.assertFalse(segment_box([2,0,.3],[3,0,.3],[-.2,-1,0],[.2,1,.6]))

    def test_route_and_events(self):
        self.assertGreaterEqual(self.validation["min_center_to_cover_m"], .30)
        self.assertGreaterEqual(self.validation["min_actor_center_separation_m"], .65)
        self.assertLessEqual(self.validation["max_route_speed_m_s"], .5)
        marks = [e for e in self.events if e["type"] == "mark"]
        self.assertEqual([e["attacker"] for e in marks], ["A","B","A"])
        self.assertEqual(self.states[-1]["hp"], [2,1])
        self.assertEqual(self.events[-1]["winner"], "A")

    def test_no_score_during_pause_or_contest(self):
        for previous, current in zip(self.states, self.states[1:]):
            if current["pause"] > 0 or current["contested"]:
                self.assertEqual(previous["score"], current["score"])

    def test_point_switch(self):
        self.assertEqual(self.states[2999]["point"],0)
        self.assertEqual(self.states[3000]["point"],1)
        self.assertEqual(self.states[3000]["scoring"],[False,False])

    def test_path_endpoints(self):
        for path in self.config["paths"].values():
            self.assertTrue(np.allclose(pose(path,0)[0],path[0][1:]))
            self.assertTrue(np.allclose(pose(path,120)[0],path[-1][1:]))


if __name__ == "__main__":
    unittest.main()
