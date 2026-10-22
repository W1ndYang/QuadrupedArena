import copy
import json
import math
from pathlib import Path
import unittest
import numpy as np
from trial_rules import Referee, cumulative_hazard, aim_valid, covers_at
from trial_motion import Planner, sweep_clear, build

CONFIG=json.loads((Path(__file__).resolve().parents[1]/"configs/trial_v3.json").read_text())


class TrialRulesTests(unittest.TestCase):
    def setUp(self):
        self.c=copy.deepcopy(CONFIG)
        self.c["arena"]["task_points"]=[dict(name="P1",center=[0,0],radius=.55),dict(name="P2",center=[4,0],radius=.55)]
        self.r=Referee(self.c);self.xy=[[0,0],[2,0]];self.yaw=[0,math.pi]

    def step(self,t,xy=None,yaw=None,covers=None):
        self.r.step(t,xy or self.xy,yaw or self.yaw,[] if covers is None else covers)

    def start(self):
        self.step(0);self.r.threshold=[1e9,1e9]

    def hit(self,actor,t):
        self.r.threshold[actor]=0.;self.step(t)

    def test_total_sixty_degree_fov(self):
        for angle,expected in ((30,True),(-30,True),(30.01,False),(-30.01,False),(180,False)):
            xy=[[0,0],[2*math.cos(math.radians(angle)),2*math.sin(math.radians(angle))]]
            self.assertEqual(aim_valid(self.c,xy,[0,0],0,[]),expected)

    def test_range(self):
        self.assertTrue(aim_valid(self.c,[[0,0],[8,0]],self.yaw,0,[]))
        self.assertFalse(aim_valid(self.c,[[0,0],[8.01,0]],self.yaw,0,[]))

    def test_three_dimensional_occlusion(self):
        wall=[dict(name="wall",center=[1,0],size=[.2,2,1.8])]
        self.assertFalse(aim_valid(self.c,self.xy,self.yaw,0,wall))
        wall[0]["size"][2]=.4
        self.assertTrue(aim_valid(self.c,self.xy,self.yaw,0,wall))

    def test_hazard_monotonic_and_humanoid_stronger(self):
        for t in (.1,.5,1,2):
            self.assertGreater(cumulative_hazard(t,1.5,.5),cumulative_hazard(t,1.5,1))
            self.assertGreater(cumulative_hazard(t+.1,1.5,1),cumulative_hazard(t,1.5,1))
        self.assertAlmostEqual(cumulative_hazard(0,1.5,1),0)

    def test_hazard_interval_composition(self):
        h=lambda t:cumulative_hazard(t,1.5,.5)
        survival=np.prod([math.exp(-(h(t+.02)-h(t))) for t in np.arange(0,1,.02)])
        self.assertAlmostEqual(survival,math.exp(-h(1)))

    def test_human_inside_does_not_contest(self):
        self.xy[1]=[.4,0];self.start();self.step(1.5)
        self.assertAlmostEqual(self.r.score,.5)

    def test_dwell_and_continuous_score(self):
        self.start();self.step(.99);self.assertEqual(self.r.score,0)
        self.step(1);self.assertEqual(self.r.score,0)
        self.step(2.5);self.assertAlmostEqual(self.r.score,1.5)

    def test_leaving_resets_dwell(self):
        self.start();self.step(.5);self.step(.52,xy=[[1,1],[2,0]])
        self.step(.54);self.step(1.5)
        self.assertEqual(self.r.score,0)

    def test_only_successful_attacker_cools(self):
        self.start();self.hit(1,.5)
        self.assertEqual(self.r.hp,[2,3]);self.assertEqual(self.r.cooldown_until,[0,2.5])
        self.assertAlmostEqual(self.r.aim[0],.5)
        self.hit(0,.52);self.assertEqual(self.r.hp,[2,2])

    def test_cooldown_prevents_aim_and_repeat_hit(self):
        self.start();self.hit(0,.5);self.step(1)
        self.assertEqual(self.r.hp,[3,2]);self.assertEqual(self.r.aim[0],0)
        self.step(2.5);self.assertEqual(self.r.aim[0],0)
        self.step(2.52);self.assertAlmostEqual(self.r.aim[0],.02)

    def test_target_invalid_even_before_dwell(self):
        self.start();self.hit(1,.2)
        self.assertEqual(self.r.eligible,[False,True]);self.assertEqual(self.r.score,0)

    def test_outside_hit_does_not_invalidate(self):
        self.xy=[[1,1],[2,1]];self.start();self.hit(1,.2)
        self.assertEqual(self.r.eligible,[True,True]);self.assertEqual(self.r.hp[0],2)

    def test_hit_preserves_earned_score_and_blocks_future(self):
        self.start();self.step(2);self.hit(1,2.02);earned=self.r.score
        self.step(3);self.assertEqual(self.r.score,earned)
        self.assertAlmostEqual(earned,1.02)

    def test_invalid_point_not_restored_by_reentry_or_b_death(self):
        self.start();self.hit(1,.2);self.r.hp[1]=0
        self.step(.4,xy=[[1,1],[2,0]]);self.step(.6);self.step(3)
        self.assertFalse(self.r.eligible[0]);self.assertEqual(self.r.score,0)

    def test_all_invalid_ends_while_a_alive(self):
        self.start();self.r.eligible[1]=False;self.hit(1,.2)
        self.assertEqual(self.r.finished,"all_targets_invalid");self.assertEqual(self.r.hp[0],2)

    def test_a_elimination_ends(self):
        self.start();self.r.hp[0]=1;self.hit(1,.2)
        self.assertEqual(self.r.finished,"quadruped_eliminated")

    def test_b_elimination_continues_scoring(self):
        self.start();self.r.hp[1]=1;self.hit(0,.2);self.step(2)
        self.assertIsNone(self.r.finished);self.assertAlmostEqual(self.r.score,1)

    def test_simultaneous_lethal_hits(self):
        self.start();self.r.hp=[1,1];self.r.threshold=[0,0];self.step(.2)
        self.assertEqual(self.r.hp,[0,0]);self.assertEqual(sum(e["type"]=="hit" for e in self.r.events),2)

    def test_occlusion_resets_continuous_aim(self):
        self.start();self.step(.5)
        self.step(.52,covers=[dict(name="wall",center=[1,0],size=[.2,2,1.8])])
        self.assertEqual(self.r.aim,[0,0]);self.step(.54);self.assertEqual(self.r.aim,[0,0])

    def test_fov_break_resets_aim(self):
        self.start();self.step(.5);self.step(.52,yaw=[math.pi,0])
        self.assertEqual(self.r.aim,[0,0])

    def test_time_limit(self):
        self.start();self.step(self.c["duration"])
        self.assertEqual(self.r.finished,"time_limit");self.assertEqual(self.r.score,39)
        with self.assertRaises(RuntimeError):self.step(41)

    def test_reproducible_separate_random_streams(self):
        r=Referee(self.c);r.step(0,self.xy,self.yaw,[]);self.step(0)
        self.assertEqual(r.threshold,self.r.threshold)
        first_b=self.r.threshold[1]
        self.r.reset_aim(0);self.step(.02)
        self.assertEqual(first_b,self.r.threshold[1])

    def test_collapse_geometry_and_los(self):
        before=covers_at(CONFIG,19.98);after=covers_at(CONFIG,20)
        xy=[[-1,3.25],[-1,0]];yaw=[-math.pi/2,math.pi/2]
        self.assertFalse(aim_valid(CONFIG,xy,yaw,1,before))
        self.assertTrue(aim_valid(CONFIG,xy,yaw,1,after))
        self.assertEqual(next(c for c in after if c["name"]=="High3")["size"],[2,1.8,.4])

    def test_collapse_sweep_detects_actor(self):
        self.assertFalse(sweep_clear(CONFIG,[[-1,1.3]]))
        self.assertTrue(sweep_clear(CONFIG,[[-1,3.25],[-4,-1.5]]))

    def test_map_change_does_not_pause_unaffected_aim(self):
        self.start();self.step(.5)
        self.r.step(.52,self.xy,self.yaw,[],map_changed=True)
        self.assertAlmostEqual(self.r.aim[0],.52)

    def test_astar_avoids_covers_and_reserved_sweep(self):
        planner=Planner(CONFIG,covers_at(CONFIG,0))
        start=np.array(CONFIG["start_a"]);route=planner.route(start,[-1,3.25])
        self.assertIsNotNone(route)
        for dest in route:
            self.assertTrue(planner.segment(start,dest));start=dest
        self.assertFalse(planner.free([-1,1.3]))

    def test_full_trial_and_render_fps_independence(self):
        states,events,validation=build(CONFIG)
        other=copy.deepcopy(CONFIG);other['fps']=30
        states2,events2,_=build(other)
        self.assertEqual(states,states2);self.assertEqual(events,events2)
        self.assertTrue(validation['collapse_sweep_checked'])
        self.assertEqual(sum(e['type']=='collapse' for e in events),1)
        self.assertGreater(validation['min_center_to_cover'],.4)
        self.assertGreater(validation['min_actor_separation'],.85)
        self.assertEqual(states[-1]['finished'],'quadruped_eliminated')
        self.assertEqual(states[-1]['score'],9.06)
        self.assertEqual(states[-1]['t'],20.14)


if __name__=="__main__":unittest.main()
