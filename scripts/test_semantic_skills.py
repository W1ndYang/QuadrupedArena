import copy
import math
import unittest
from unittest.mock import patch
import numpy as np
from adaptive_motion import load_config
from llm_trial import PROJECT
from vision_core import VisionMatch,public_packet
from semantic_skills import SemanticSkills,VisualMemory,ground_point,SKILLS


def frame(t=0):return dict(file='test.png',simulation_seconds=t,camera_world_yaw_degrees=0,camera_eye=[-5,-2.8,.6])
def pixel(xy,f):
    yaw=math.radians(f['camera_world_yaw_degrees']);p=math.radians(-5);d=np.asarray([*xy,0])-f['camera_eye']
    forward=np.array([math.cos(yaw)*math.cos(p),math.sin(yaw)*math.cos(p),math.sin(p)]);right=np.array([math.sin(yaw),-math.cos(yaw),0]);up=np.cross(right,forward)
    return [float(384+384*np.dot(d,right)/np.dot(d,forward)),float(216-384*np.dot(d,up)/np.dot(d,forward))]


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config(PROJECT/'configs/trial_rotating.json');self.match=VisionMatch(self.cfg)
        self.packet=public_packet(0,self.match.states[-1],[frame()],self.cfg['start_b']);self.engine=SemanticSkills(self.cfg)
    def request(self,skill,**params):return dict(observation_id=0,skill=skill,params=params,reason='测试')
    def test_projection_roundtrip(self):
        for xy in ([-1,-2.8],[0,-1],[-2,-3]):self.assertTrue(np.allclose(ground_point(frame(),pixel(xy,frame())),xy))
    def test_bad_pixels(self):
        for p in ([0,0],[999,200],[float('nan'),200],[True,200]):
            with self.assertRaises(ValueError):ground_point(frame(),p)
    def test_no_future_sighting(self):
        with self.assertRaises(ValueError):self.engine.validate(self.request('AIM_AT_TRACK',sighting=dict(time=2,feet=[384,250])),self.packet)
    def test_reject_path_and_long_reason(self):
        for r in (self.request('MOVE_TO',goal=[1,2]),self.request('MOVE_TO',target='enemy'),dict(self.request('OBSERVE'),reason='a'*101)):
            with self.assertRaises(ValueError):self.engine.validate(r,self.packet)
    def test_all_five_skills(self):
        for skill in SKILLS:
            e=SemanticSkills(self.cfg);action,route,cam,status=e.commit(self.request(skill),self.packet)
            self.assertNotEqual(action['skill'],'AIM');self.assertNotEqual(action['facing'],'enemy')
            self.assertIn('state',status);previous=np.asarray(self.packet['self_state']['xy'])
            for dest in route:self.assertTrue(e.base.segment(previous,dest));previous=dest
    def test_capture_inside_circle(self):
        _,_,_,status=self.engine.commit(self.request('CAPTURE_POINT',target='P1'),self.packet)
        self.assertLess(np.linalg.norm(np.asarray(status['goal'])-[-4,-1.5]),.55)
        self.assertGreater(status['estimated_arrival_seconds'],0)
    def test_expired_target_interrupt(self):
        self.engine.commit(self.request('CAPTURE_POINT'),self.packet);p=copy.deepcopy(self.packet);p['public_state']['target_generation']=2
        action,route=self.engine.tick(p);self.assertEqual(action['skill'],'STOP');self.assertEqual(route,[])
    def test_hit_interrupt(self):
        self.engine.commit(self.request('CAPTURE_POINT'),self.packet);p=copy.deepcopy(self.packet);p['self_state']['hp']=2
        action,route=self.engine.tick(p);self.assertTrue(self.engine.status['state'].startswith('EMERGENCY_'))
        self.assertEqual(self.engine.status['interrupted_skill'],'CAPTURE_POINT');self.assertEqual(self.engine.intent,'TAKE_COVER')
    def test_aim_needs_visual_not_initial(self):
        a,_,_,s=self.engine.commit(self.request('AIM_AT_TRACK'),self.packet);self.assertFalse(a['attack']);self.assertEqual(s['state'],'TARGET_STALE')
    def test_visual_aim_expiry(self):
        a,_,_,_=self.engine.commit(self.request('AIM_AT_TRACK',sighting=dict(time=0,feet=pixel([-1,-2.8],frame()))),self.packet)
        self.assertTrue(a['attack']);p=copy.deepcopy(self.packet);p['simulation_seconds']=2.02;self.engine.tick(p)
        self.assertFalse(self.engine.action['attack']);self.assertEqual(self.engine.status['state'],'TARGET_STALE')
        self.assertIsNone(self.engine.memory.belief(4.01))
    def test_hidden_mutation_invariance(self):
        s=copy.deepcopy(self.match.states[-1]);s['xy'][1]=[-999,999];s['hp'][1]=0;s['human_state']='secret';s['yaw'][1]=17
        p=public_packet(0,s,[frame()],self.cfg['start_b'])
        a=SemanticSkills(self.cfg).commit(self.request('TAKE_COVER'),self.packet)
        b=SemanticSkills(self.cfg).commit(self.request('TAKE_COVER'),p)
        self.assertEqual(a[0],b[0]);self.assertEqual(a[3],b[3]);self.assertNotIn('human_patrol',self.engine.cfg['policy'])
    def test_turn_time_in_eta(self):
        own=dict(xy=[0,0],yaw=math.pi,speed=0)
        front=self.engine.eta(own,[np.array([1.,0])],False);reverse=self.engine.eta(own,[np.array([1.,0])],True)
        self.assertGreater(front,reverse)
    def test_no_world_position_annotation(self):
        with self.assertRaises(ValueError):self.engine.validate(self.request('AIM_AT_TRACK',sighting=dict(time=0,xy=[1,2])),self.packet)
    def test_observe_direction_not_overridden_by_memory(self):
        self.engine.commit(self.request('OBSERVE',direction='north'),self.packet)
        self.engine.tick(self.packet);self.assertAlmostEqual(self.engine.action['facing'],math.pi/2)
    def test_observe_camera_not_forced_to_visual_track(self):
        self.engine.commit(self.request('OBSERVE',direction='north',sighting=dict(time=0,feet=pixel([-1,-2.8],frame()))),self.packet)
        self.engine.tick(self.packet);self.assertAlmostEqual(self.engine.action['facing'],math.pi/2)
        self.assertEqual(self.engine.camera(self.packet)['mode'],'sweep')
    def test_unsafe_route_rejected(self):
        self.packet['self_state']['hp']=1;self.packet['public_state']['active_target']=1
        with patch('semantic_skills.BeliefPlanner.point_risk',return_value=1.):
            a,r,c,s=self.engine.commit(self.request('CAPTURE_POINT',target='P2'),self.packet)
        self.assertEqual(s['state'],'UNSAFE_ROUTE');self.assertEqual(a['skill'],'HOLD');self.assertEqual(r,[])
    def test_expired_capture_map_replan_does_not_select_next_point(self):
        self.engine.commit(self.request('CAPTURE_POINT'),self.packet);p=copy.deepcopy(self.packet);p['public_state']['target_generation']=2;p['public_state']['active_target']=1
        a,r,c,s=self.engine.plan(p);self.assertEqual(s['state'],'TARGET_EXPIRED');self.assertEqual(a['skill'],'STOP')
    def test_stationary_eta_zero(self):
        self.assertEqual(self.engine.eta(self.packet['self_state'],[np.asarray(self.packet['self_state']['xy'])],False),0)


if __name__=='__main__':unittest.main()
