import copy
import json
import math
from pathlib import Path
import pickle
import unittest
import numpy as np
from llm_trial import Match
from adaptive_motion import load_config


class LLMInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config(Path(__file__).resolve().parents[1]/'configs/trial_rotating.json')
        self.cfg['seed']=12345;self.cfg['lambda_max']=0.
        self.m=Match(self.cfg)

    def run_action(self,action):
        route=self.m.compile(action);self.m.apply(action,route);self.m.advance()

    def test_fixed_window(self):
        self.run_action(dict(skill='HOLD'))
        self.assertEqual(self.m.t,2);self.assertEqual(len(self.m.states),101)

    def test_observe_does_not_advance(self):
        before=pickle.dumps(self.m)
        for _ in range(3):self.m.observe()
        self.assertEqual(before,pickle.dumps(self.m))

    def test_no_rng_in_observation(self):
        def keys(obj):
            if isinstance(obj,dict):return set(obj)|set().union(*(keys(v) for v in obj.values()))
            if isinstance(obj,list):return set().union(*(keys(v) for v in obj))
            return set()
        self.assertFalse(keys(self.m.observe())&{'threshold','rng','seed','hazard'})

    def test_invalid_goal_rejected_without_changing_state(self):
        before=pickle.dumps(self.m)
        for goal in ([10,0],[-3,-1.5],[float('nan'),0]):
            with self.assertRaises(ValueError):self.m.compile(dict(skill='MOVE',goal=goal))
        self.assertEqual(before,pickle.dumps(self.m))

    def test_skill_whitelist_and_speed_limit(self):
        for action in (dict(skill='PYTHON'),dict(skill='MOVE'),dict(skill='HOLD',speed_limit=5),dict(skill='HOLD',facing='teleport')):
            with self.assertRaises(ValueError):self.m.compile(action)

    def test_no_instant_turn_or_teleport(self):
        self.run_action(dict(skill='MOVE',goal=[-4,-1.5]))
        for a,b in zip(self.m.states,self.m.states[1:]):
            self.assertLessEqual(np.linalg.norm(np.array(b['xy'][0])-a['xy'][0]),1.1*.02+1e-8)
            self.assertLessEqual(abs(b['yaw'][0]-a['yaw'][0]),2.5*.02+1e-8)

    def test_resume_deterministically(self):
        self.run_action(dict(skill='MOVE',goal=[-4,-1.5]))
        other=pickle.loads(pickle.dumps(self.m))
        action=dict(skill='HOLD',facing='keep');self.run_action(action)
        other.apply(action,other.compile(action));other.advance()
        self.assertEqual(self.m.states,other.states)

    def test_generation_bound_skill_stops_when_expired(self):
        self.cfg['start_a']=[-4,-1.5];self.m=Match(self.cfg)
        self.run_action(dict(skill='HOLD',target_generation=1))
        self.m.advance();self.m.advance()
        self.assertEqual(self.m.ref.generation,2);self.assertEqual(self.m.active['skill'],'STOP')
        self.assertTrue(any(e.get('reason')=='target_expired' for e in self.m.events))

    def test_no_automatic_tactical_evasion(self):
        self.run_action(dict(skill='HOLD',facing=0))
        self.assertFalse(self.m.a.route)
        self.assertNotIn('QuadrupedPolicy',type(self.m).__dict__)


if __name__=='__main__':unittest.main()
