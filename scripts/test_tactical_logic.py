"""CPU-only contract tests: perception, rules, routes and wheel animation."""
import json
import math
from pathlib import Path
import unittest
import numpy as np
from tactical_logic import build, block, observe, Mover, CoverController

class TacticalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        cls.config=json.loads((root/"configs/demo_tactical_v2.json").read_text())
        cls.arena=json.loads((root/"configs/arena_v0.json").read_text())
        cls.states,cls.events,cls.validation=build(cls.config,cls.arena)

    def test_cover_and_fov(self):
        wall=[{"name":"wall","center":[0,0],"size":[.4,2,.6]}]
        hit=block([-1,0,.3],[1,0,.3],wall)
        self.assertTrue(np.allclose(hit[2],[-.2,0,.3]))
        self.assertIsNone(block([-1,0,1],[1,0,1],wall))
        self.assertIsNone(observe(np.array([-1,0]),math.pi,1,[1,0],1,wall))
        self.assertIsNotNone(observe(np.array([-1,0]),0,1,[1,0],1,wall))

    def test_full_sequence_and_clearance(self):
        self.assertEqual(self.validation["states"], ["TAKE_COVER","HIDDEN","PEEK","AIM","WITHDRAW","WATCH","RELOCATE","NEW_COVER","TO_POINT","CAPTURE"])
        self.assertGreater(self.validation["min_center_cover_distance"],.40)
        self.assertGreater(self.validation["min_actor_separation"],.7)
        self.assertLessEqual(self.validation["max_speed_a"],1.4)
        self.assertTrue(any(s["scoring"][0] for s in self.states))
        for s in self.states:
            if s["state"] in ("HIDDEN","NEW_COVER"):
                self.assertFalse(s["los"],s["t"])

    def test_hidden_memory_does_not_track_truth(self):
        m=Mover([-3.85,-1.6],0,1,2,3)
        c=CoverController(self.config,self.arena,m)
        c.update(0,.02,np.array([-.9,-1.9]),False,0)
        remembered=c.memory.copy()
        c.update(.02,.02,None,False,0)
        self.assertTrue(np.array_equal(c.memory,remembered))
        self.assertGreater(c.memory_age,0)

    def test_no_flank_means_no_relocation(self):
        config=dict(self.config,opponent_flank=False)
        _,events,_=build(config,self.arena)
        self.assertFalse(any(e.get("state")=="RELOCATE" for e in events))

    def test_lock_and_scoring_rules(self):
        for prev,s in zip(self.states,self.states[1:]):
            if not s["los"]:
                self.assertEqual(s["lock"],[0.,0.])
            if s["pause"]>0 or s["contested"]:
                self.assertEqual(s["score"],prev["score"])
        for e in self.events:
            if e["type"]=="mark":
                self.assertTrue(e["clear_los"])
                i=round(e["time"]/.02); side="AB".index(e["attacker"])
                self.assertGreaterEqual(self.states[i-1]["lock"][side],.78-1e-7)
        self.assertEqual(self.events[-1]["winner"],"A")

    def test_wheel_rotation_matches_translation(self):
        m=Mover([0,0],0,1,2,3); m.navigate([[10,0]])
        for _ in range(100): m.step(.02)
        self.assertTrue(np.allclose(m.wheels,m.distance/.089))
        self.assertAlmostEqual(m.xy[0],m.distance)

if __name__=="__main__": unittest.main()
