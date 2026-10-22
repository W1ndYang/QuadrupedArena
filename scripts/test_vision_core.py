import copy
import unittest
from adaptive_motion import load_config
from llm_trial import PROJECT
from vision_core import CameraRig,VisionMatch,public_packet


class VisionTests(unittest.TestCase):
    def setUp(self):self.match=VisionMatch(load_config(PROJECT/'configs/trial_rotating.json'))

    def test_no_hidden_fields(self):
        s=copy.deepcopy(self.match.states[-1]);one=public_packet(1,s,[],[5,2.8])
        s['xy'][1]=[999,999];s['yaw'][1]=999;s['hp'][1]=0;s['cooldown'][1]=999
        s['human_state']='secret';s['human_memory']=[999,999];s['aim']=[999,999];s['human_visible']=True
        self.assertEqual(one,public_packet(1,s,[],[-999,999]))

    def test_initial_only(self):
        s=self.match.states[-1]
        self.assertIn('opponent_initial_xy',public_packet(0,s,[],[5,2.8]))
        self.assertNotIn('opponent_initial_xy',public_packet(1,s,[],[5,2.8]))

    def test_reject_truth_tracking(self):
        for a in ({'skill':'AIM'},{'skill':'HOLD','facing':'enemy'}):
            with self.assertRaises(ValueError):self.match.compile(a)

    def test_sweep_and_rate(self):
        r=CameraRig();c=dict(mode='sweep',center_degrees=0,amplitude_degrees=60);samples=[]
        for i in range(1,101):
            old=r.pan;r.step(c,i*.02,.02);self.assertLessEqual(abs(r.pan-old),2.4+1e-8)
            if i%25==0:samples.append(r.pan)
        for a,b in zip(samples,[60,0,-60,0]):self.assertAlmostEqual(a,b)

    def test_hold_no_teleport(self):
        r=CameraRig();self.assertAlmostEqual(r.step(dict(mode='hold',yaw_degrees=90),0,.02),2.4)

    def test_invalid_rig(self):
        for c in ({'mode':'enemy'},{'mode':'hold','yaw_degrees':180},{'mode':'sweep','center_degrees':80,'amplitude_degrees':60}):
            with self.assertRaises(ValueError):CameraRig.validate(c)


if __name__=='__main__':unittest.main()
