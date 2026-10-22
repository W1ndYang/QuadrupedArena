"""R0 v0.4 lifecycle boundaries, backward compatibility and fast scanning."""
import copy
import math
from pathlib import Path
import unittest
import numpy as np
from adaptive_motion import load_config, build, HumanPolicy
from tactical_logic import Mover, wrap
from trial_motion import Planner
from trial_rules import Referee, covers_at

CONFIG=load_config(Path(__file__).resolve().parents[1]/'configs/trial_rotating.json')


class RotatingRulesTests(unittest.TestCase):
    def setUp(self):
        self.cfg=copy.deepcopy(CONFIG)
        self.cfg['arena']['task_points']=[dict(name='P1',center=[0,0],radius=.55),dict(name='P2',center=[4,0],radius=.55)]
        self.r=Referee(self.cfg)

    def step(self,t,point=(0,0),attacks=False):
        self.r.step(t,[list(point),[2,0]],[0,math.pi],[],attack_enabled=[attacks,attacks])

    def test_only_first_site_initially_active(self):
        self.assertEqual(self.r.eligible,[True,False]);self.assertEqual(self.r.generation,1)

    def test_untriggered_site_does_not_expire(self):
        self.step(0,(-2,0));self.step(12,(-2,0))
        self.assertIsNone(self.r.expires_at);self.assertEqual(self.r.active_target,0)

    def test_inactive_site_neither_triggers_nor_scores(self):
        self.step(0,(4,0));self.step(6,(4,0))
        self.assertEqual(self.r.score,0);self.assertIsNone(self.r.triggered_at)

    def test_timer_begins_on_first_entry_not_dwell_completion(self):
        self.step(0,(-2,0));self.step(2)
        self.assertEqual(self.r.expires_at,7);self.assertEqual(self.r.score,0)

    def test_expiry_and_maximum_four_points(self):
        self.step(0);self.step(4.98)
        self.assertAlmostEqual(self.r.score,3.98);self.assertEqual(self.r.active_target,0)
        self.step(5);self.assertEqual(self.r.score,4)
        self.assertEqual(self.r.eligible,[False,True]);self.assertIsNone(self.r.finished)
        self.assertEqual(self.r.snapshot(5)['target_generation'],2)

    def test_large_step_caps_score_at_expiry(self):
        self.step(0);self.step(8)
        self.assertEqual(self.r.score,4)
        expired=next(e for e in self.r.events if e['type']=='target_invalidated')
        self.assertEqual(expired['time'],5);self.assertEqual(expired['reason'],'timeout')

    def test_leaving_and_reentry_do_not_reset_timer(self):
        self.step(0);self.step(.4,(-2,0));self.step(2)
        self.assertEqual(self.r.expires_at,5);self.assertEqual(self.r.dwell[0],0)
        self.step(5);self.assertEqual(self.r.score,2)

    def test_cyclic_refresh_resets_instance_not_total_score(self):
        self.step(0);self.step(5);self.step(6,(4,0));self.step(11,(4,0))
        self.assertEqual(self.r.active_target,0);self.assertEqual(self.r.generation,3)
        self.assertEqual(self.r.score,8);self.assertEqual(self.r.dwell,[0,0])
        self.step(12);self.step(17)
        self.assertEqual(self.r.active_target,1);self.assertEqual(self.r.generation,4)
        self.assertEqual(self.r.score,12)

    def test_hit_refreshes_immediately_without_free_points(self):
        self.step(0,attacks=True);self.r.threshold=[1e9,0]
        self.step(.2,attacks=True)
        self.assertEqual(self.r.hp[0],2);self.assertEqual(self.r.active_target,1)
        self.assertEqual(self.r.score,0);self.assertIsNone(self.r.finished)
        self.assertEqual(next(e for e in self.r.events if e['type']=='target_invalidated')['reason'],'hit')

    def test_timeout_and_hit_same_tick_do_not_double_refresh(self):
        self.step(0,attacks=True);self.r.threshold=[1e9,0]
        self.step(5,attacks=True)
        self.assertEqual(self.r.generation,2);self.assertEqual(self.r.hp[0],2)
        self.assertEqual(len([e for e in self.r.events if e['type']=='target_invalidated']),1)

    def test_lethal_hit_does_not_spawn_next_target(self):
        self.step(0,attacks=True);self.r.hp[0]=1;self.r.threshold=[1e9,0]
        self.step(.2,attacks=True)
        self.assertEqual(self.r.finished,'quadruped_eliminated');self.assertEqual(self.r.generation,1)

    def test_time_limit_wins_over_refresh(self):
        self.cfg['duration']=5;self.r=Referee(self.cfg)
        self.step(0);self.step(5)
        self.assertEqual(self.r.finished,'time_limit');self.assertEqual(self.r.score,4)
        self.assertEqual(self.r.generation,1)

    def test_human_death_does_not_stop_refresh(self):
        self.step(0);self.r.hp[1]=0;self.step(5)
        self.assertEqual(self.r.generation,2);self.assertIsNone(self.r.finished)

    def test_invalid_lifetime_rejected(self):
        for value in (0,-1,float('nan'),float('inf')):
            self.cfg['target_rotation']['lifetime_seconds']=value
            with self.assertRaises(ValueError):Referee(self.cfg)


class FastHumanTests(unittest.TestCase):
    def test_recursive_config_preserves_sensor_and_risk_parameters(self):
        self.assertEqual(CONFIG['policy']['human_sensor_half_angle_degrees'],60)
        self.assertEqual(CONFIG['policy']['risk_weight'],7)
        self.assertEqual(CONFIG['yaw_rate'],[2.5,3.0])

    def test_patrol_scan_uses_requested_rate_with_bounded_turn(self):
        cfg=copy.deepcopy(CONFIG);cfg['policy']['human_patrol']=[[0,-.1]]
        policy=HumanPolicy(cfg);planner=Planner(cfg,covers_at(cfg,0))
        mover=Mover([0,-.1],0,.65,1.2,cfg['yaw_rate'][1])
        for i in range(50):
            policy.update(i*.02,mover,None,planner);mover.step(.02,policy.look)
        self.assertAlmostEqual(mover.yaw,2.4)

    def test_last_seen_scan_uses_requested_rate(self):
        policy=HumanPolicy(CONFIG);planner=Planner(CONFIG,covers_at(CONFIG,0))
        mover=Mover([0,-.1],0,.65,1.2,3)
        policy.update(0,mover,dict(xy=[0,-.1],time=0),planner)
        policy.update(.02,mover,None,planner);mover.step(.02,policy.look)
        self.assertEqual(policy.state,'SEARCH_SCAN');self.assertAlmostEqual(mover.yaw,.048)


class RotatingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.states,cls.events,cls.validation=build(CONFIG)

    def test_refresh_occurs_and_controller_switches(self):
        self.assertTrue(any(e['type']=='target_invalidated' and e.get('reason')=='timeout' for e in self.events))
        self.assertTrue(any(e['type']=='select_target' and e['point']==1 for e in self.events))
        for before,after in zip(self.states,self.states[1:]):
            if not before['finished']:self.assertEqual(after['selected'],before['active_target'])

    def test_timer_score_and_turn_invariants(self):
        for state in self.states:
            self.assertLessEqual(state['target_instance_score'],4+1e-6)
            if state['hp'][0]>0:self.assertEqual(sum(state['eligible']),1)
            if state['target_remaining'] is not None:
                self.assertGreaterEqual(state['target_remaining'],0)
                self.assertLessEqual(state['target_remaining'],5)
        for before,after in zip(self.states,self.states[1:]):
            for i in range(2):self.assertLessEqual(abs(wrap(after['yaw'][i]-before['yaw'][i])),CONFIG['yaw_rate'][i]*CONFIG['dt']+1e-6)
        self.assertIn(self.states[-1]['finished'],('time_limit','quadruped_eliminated'))


if __name__=='__main__':unittest.main()
