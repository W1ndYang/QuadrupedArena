import copy
import inspect
import math
import unittest
import numpy as np
from adaptive_motion import (load_config,visible_observation,HumanPolicy,QuadrupedPolicy,
    RiskPlanner,ReversibleMover,exposure,build)
from tactical_logic import Mover,wrap
from trial_motion import Planner
from trial_rules import Referee,aim_valid,covers_at

CONFIG=load_config()


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=copy.deepcopy(CONFIG);self.planner=Planner(self.cfg,covers_at(self.cfg,0))
        self.b=Mover([0,-.1],0,.65,1.2,1.2);self.policy=HumanPolicy(self.cfg)

    def test_hidden_target_never_observed(self):
        wall=[dict(name='wall',center=[0,0],size=[.4,2,1.8])]
        self.assertIsNone(visible_observation(self.cfg,[-2,0],0,[2,0],wall,0))

    def test_sensor_fov_and_range(self):
        for degrees,expected in ((0,True),(59,True),(61,False),(180,False)):
            point=np.array([math.cos(math.radians(degrees)),math.sin(math.radians(degrees))])*2
            self.assertEqual(visible_observation(self.cfg,[0,0],0,point,[],0) is not None,expected)
        self.assertIsNone(visible_observation(self.cfg,[0,0],0,[8.1,0],[],0))

    def test_detection_cone_not_shooting_cone(self):
        target=[2*math.cos(math.pi/4),2*math.sin(math.pi/4)]
        self.assertIsNotNone(visible_observation(self.cfg,[0,0],0,target,[],0))
        self.assertFalse(aim_valid(self.cfg,[target,[0,0]],[0,0],1,[]))

    def test_memory_is_a_copy(self):
        position=[1,0];obs=dict(time=0,xy=position)
        self.policy.update(0,self.b,obs,self.planner)
        position[0]=99
        self.assertEqual(self.policy.memory.tolist(),[1,0])

    def test_lost_target_keeps_last_observation_not_truth(self):
        self.policy.update(0,self.b,dict(time=0,xy=[1,0]),self.planner)
        for t in (.1,1,2):self.policy.update(t,self.b,None,self.planner)
        self.assertEqual(self.policy.memory.tolist(),[1,0]);self.assertEqual(self.policy.seen_at,0)
        self.assertEqual(self.policy.state,'SEARCH_LAST_SEEN')

    def test_memory_expires(self):
        self.policy.update(0,self.b,dict(time=0,xy=[1,0]),self.planner)
        self.policy.update(4.01,self.b,None,self.planner)
        self.assertIsNone(self.policy.memory);self.assertIsNone(self.policy.seen_at)
        self.assertIn(self.policy.state,('PATROL','PATROL_SCAN'))

    def test_no_true_target_argument_on_human_interface(self):
        self.assertEqual(list(inspect.signature(HumanPolicy.update).parameters),
            ['self','t','mover','observation','planner','own_cooldown_until','alive'])

    def test_hidden_world_variants_produce_identical_human_decisions(self):
        # Different unseen target positions, same B state and observation history.
        wall=[dict(name='wall',center=[0,0],size=[.4,6,1.8])]
        output=[]
        for hidden in ([2,0],[2,2]):
            b=Mover([-2,0],0,.65,1.2,1.2);policy=HumanPolicy(self.cfg)
            obs=visible_observation(self.cfg,b.xy,b.yaw,hidden,wall,0)
            self.assertIsNone(obs);policy.update(0,b,obs,self.planner)
            output.append((policy.state,policy.look,[p.tolist() for p in b.route],policy.memory))
        self.assertEqual(output[0],output[1])

    def test_cooldown_does_not_reveal_hidden_target(self):
        self.policy.update(0,self.b,None,self.planner,own_cooldown_until=2)
        self.assertFalse(self.policy.visible);self.assertIsNone(self.policy.memory)
        self.assertIn(self.policy.state,('PATROL','PATROL_SCAN'))

    def test_referee_cannot_autofire_without_acquisition(self):
        r=Referee(self.cfg);xy=[[0,0],[2,0]];yaw=[0,math.pi]
        r.step(0,xy,yaw,[],attack_enabled=[False,False]);r.threshold=[0,0]
        r.step(.02,xy,yaw,[],attack_enabled=[False,False])
        self.assertEqual(r.hp,[3,3]);self.assertFalse(any(e['type']=='hit' for e in r.events))


class RiskPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg=copy.deepcopy(CONFIG);self.planner=Planner(self.cfg,covers_at(self.cfg,0))

    def test_cover_reduces_risk_to_zero(self):
        walls=[dict(name='wall',center=[0,0],size=[.4,2,1.8])]
        self.assertEqual(exposure(self.cfg,[-2,0],np.array([2,0]),math.pi,walls),0)
        self.assertGreater(exposure(self.cfg,[-2,0],np.array([2,0]),math.pi,[]),1)

    def test_forward_fire_more_expensive_than_back(self):
        self.assertGreater(exposure(self.cfg,[2,0],np.array([0,0]),0,[]),exposure(self.cfg,[2,0],np.array([0,0]),math.pi,[]))

    def test_dead_enemy_has_no_risk_or_dynamic_exclusion(self):
        p=RiskPlanner(self.planner,[0,-.1],0,3,enemy_alive=False)
        self.assertEqual(sum(p.risk.values()),0);self.assertTrue(p.dynamic_free([0,-.1]))

    def test_low_hp_increases_risk_weight(self):
        a=RiskPlanner(self.planner,[0,-.1],0,3);b=RiskPlanner(self.planner,[0,-.1],0,1)
        self.assertGreater(b.weight,a.weight)

    def test_public_hit_opens_cooldown_window(self):
        a=QuadrupedPolicy(self.cfg)
        a.public_hit(dict(time=5,type='hit',attacker=0));self.assertEqual(a.known_b_cooldown,0)
        a.public_hit(dict(time=5,type='hit',attacker=1));self.assertEqual(a.known_b_cooldown,7)

    def test_cooldown_reduces_not_erases_path_risk(self):
        a=RiskPlanner(self.planner,[0,-.1],0,3);b=RiskPlanner(self.planner,[0,-.1],0,3,True)
        self.assertGreater(b.weight,0);self.assertLess(b.weight,a.weight)

    def test_dynamic_enemy_is_excluded(self):
        p=RiskPlanner(self.planner,[-1,3.25],0,3)
        self.assertFalse(p.free([-1,3.25]));p.solve([-4,-1.5])
        route=p.target_route(self.cfg['arena']['task_points'][1])
        self.assertIsNone(route)  # Entire small circle blocked by the dynamic safety radius.

    def test_risk_route_clearance_and_cost(self):
        start=np.array([-4,-1.5]);target=np.array([-1,3.25]);enemy=np.array([-2.7,3.2])
        p=RiskPlanner(self.planner,enemy,-math.pi/2,2);p.solve(start);route=p.route(target)
        self.assertIsNotNone(route)
        previous=start
        for waypoint in route:
            self.assertTrue(p.segment(previous,waypoint));previous=waypoint
        self.assertTrue(np.allclose(route[-1],target))

    def test_sustained_threat_triggers_evasion(self):
        a=QuadrupedPolicy(self.cfg);m=Mover([-4,-1.5],0,1.1,2,2.5)
        enemy=np.array([-4,-3.2]);yaw=math.pi/2
        a.update(0,m,enemy,yaw,True,[True,True],3,self.planner)
        a.update(.3,m,enemy,yaw,True,[True,True],3,self.planner)
        self.assertEqual(a.state,'EVADE');self.assertTrue(m.route)
        self.assertEqual(exposure(self.cfg,m.route[-1],enemy,yaw,self.planner.covers),0.)

    def test_hidden_safe_hold_scans_instead_of_tracking_truth(self):
        a=QuadrupedPolicy(self.cfg);a.selected=0;m=Mover([-4,-1.5],0,1.1,2,2.5)
        a.update(0,m,[2,-1.5],math.pi,True,[True,True],3,self.planner)
        self.assertEqual(a.state,'HOLD');self.assertAlmostEqual(a.look,.6*self.cfg['dt'])

    def test_entering_circle_does_not_cancel_route_to_protected_interior(self):
        a=QuadrupedPolicy(self.cfg);a.selected=0;a.last_plan=0
        m=Mover([-4,-1.95],math.pi/2,1.1,2,2.5);m.navigate([[-4,-1.5]])
        a.update(.02,m,[2,-1.5],math.pi,True,[True,True],3,self.planner)
        self.assertTrue(m.route);self.assertNotEqual(a.state,'HOLD')

    def test_reverse_evasion_does_not_require_half_turn(self):
        m=ReversibleMover([0,0],0,1.1,2,2.5);m.reverse_allowed=True;m.navigate([[-2,0]])
        for _ in range(50):m.step(.02)
        self.assertLess(m.xy[0],-.5);self.assertAlmostEqual(m.yaw,0)
        self.assertGreaterEqual(m.v,-.8);self.assertTrue(all(v<0 for v in m.wheels))

    def test_normal_advance_does_not_enable_reverse(self):
        m=ReversibleMover([0,0],0,1.1,2,2.5);m.navigate([[-2,0]]);m.step(.02)
        self.assertEqual(m.v,0);self.assertAlmostEqual(abs(m.yaw),.05)

    def test_risk_route_reduces_exposure_weighted_cost(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['collapse']['planner_reserved_sweep']=dict(name='unused',center=[20,20],size=[1,1,1])
        wall=dict(name='wall',center=[0,0],size=[.4,3,1.8]);base=Planner(cfg,[wall])
        start=np.array([-4,0]);p=RiskPlanner(base,[0,3],-math.pi/2,2);p.solve(start);route=p.route([4,0])
        shortest=base.route(start,[4,0]);self.assertIsNotNone(route)
        cost=lambda path:sum(p.segment_cost(a,b) for a,b in zip([start]+path,path))
        # The preferred side depends on BOTH LOS and orientation. It need not
        # be the visually "back" side of a thin wall from every endpoint.
        self.assertLess(cost(route),cost(shortest))


class AdaptiveIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.states,cls.events,cls.validation=build(CONFIG)

    def test_real_decision_transitions(self):
        self.assertTrue(any(s['a_state']=='EVADE' for s in self.states))
        self.assertTrue(any(s['a_state']=='WINDOW_ADVANCE' for s in self.states))
        self.assertTrue(any(s['human_state']=='SEARCH_LAST_SEEN' for s in self.states))
        self.assertTrue(self.validation['collapse_sweep_checked'])

    def test_no_hidden_memory_tracking(self):
        for before,now in zip(self.states,self.states[1:]):
            if not now['human_visible'] and now['human_memory'] is not None:
                self.assertEqual(now['human_memory'],before['human_memory'])
                self.assertGreater(now['human_memory_age'],before['human_memory_age'])

    def test_human_shots_require_sensor_and_referee_geometry(self):
        for e in self.events:
            if e['type']=='hit' and e['attacker']==1:
                state=self.states[round(e['time']/CONFIG['dt'])]
                self.assertTrue(state['human_visible'])
                self.assertTrue(aim_valid(CONFIG,state['xy'],state['yaw'],1,covers_at(CONFIG,e['time'])))

    def test_no_instant_turns_or_overlaps(self):
        self.assertGreaterEqual(self.validation['min_center_to_cover'],CONFIG['planner_clearance']-1e-6)
        self.assertGreaterEqual(self.validation['min_actor_separation'],CONFIG['policy']['safety_stop_distance']-1e-6)
        for before,now in zip(self.states,self.states[1:]):
            for i in range(2):self.assertLessEqual(abs(wrap(now['yaw'][i]-before['yaw'][i])),CONFIG['yaw_rate'][i]*CONFIG['dt']+1e-6)


if __name__=='__main__':unittest.main()
