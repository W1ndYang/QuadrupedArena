import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np

from rl_skill_env import ACTION_NAMES,NativeGeometrySensor,SkillEnv,load_rl_config
from rl_skill_ppo import PPOAgent,compute_gae
from semantic_skills import VisualMemory


class SkillEnvTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_rl_config();self.cfg['sensor']['dropout_probability']=0.;self.cfg['sensor']['position_noise_std_m']=0.

    def test_contract_and_10hz_clock(self):
        env=SkillEnv(self.cfg,seed=3);observation,info=env.reset(3)
        self.assertEqual(observation.shape,(env.observation_size,));self.assertEqual(env.action_size,len(ACTION_NAMES))
        after,_,done,truncated,step=env.step(0)
        self.assertAlmostEqual(step['time'],.1);self.assertFalse(done);self.assertFalse(truncated)
        self.assertEqual(after.shape,observation.shape);self.assertEqual(step['sensor_last_sample_time'],.1)
        self.assertEqual(float(np.sum(observation[-env.action_size:])),0.)

    def test_same_intent_continues_without_recommit(self):
        env=SkillEnv(self.cfg,seed=4);env.reset(4)
        _,_,_,_,first=env.step(1);sequence=env.engine.sequence
        _,_,_,_,second=env.step(1)
        self.assertTrue(first['feedback']['committed']);self.assertTrue(second['feedback']['continued'])
        self.assertEqual(env.engine.sequence,sequence)

    def test_actor_observation_invariant_to_hidden_opponent_truth(self):
        a=SkillEnv(self.cfg,seed=5);a.reset(5);a.engine.memory.track=dict(xy=[1,1],seen_at=0,source='simulator_native_geometry',base_uncertainty=.2)
        before=a._observation();a.match.states[-1]['xy'][1]=[999,999];a.match.states[-1]['yaw'][1]=99
        a.match.states[-1]['hp'][1]=0;a.match.states[-1]['human_state']='SECRET'
        self.assertTrue(np.array_equal(before,a._observation()))

    def test_simulator_memory_is_explicit_and_validated(self):
        memory=VisualMemory([5,2.8]);memory.update_simulator_sensor(dict(time=.1,xy=[4,2],uncertainty=.2))
        belief=memory.belief(.1);self.assertEqual(belief['source'],'simulator_native_geometry');self.assertTrue(belief['aim_usable'])
        for bad in (dict(time=.1,xy=[4,2]),dict(time=.1,xy=[4],uncertainty=.2),dict(time=.1,xy=[4,2],uncertainty=-1)):
            with self.assertRaises(ValueError):memory.update_simulator_sensor(bad)

    def test_deterministic_seed(self):
        outputs=[]
        for _ in range(2):
            env=SkillEnv(seed=12);obs,_=env.reset(12);trace=[]
            for action in (5,1,1,0,4,0):
                obs,reward,done,_,info=env.step(action);trace.append((obs.copy(),reward,info['official_score']))
            outputs.append(trace)
        for x,y in zip(*outputs):
            self.assertTrue(np.array_equal(x[0],y[0]));self.assertEqual(x[1:],y[1:])


class PPOTests(unittest.TestCase):
    def test_update_checkpoint_and_inference(self):
        rng=np.random.default_rng(9);agent=PPOAgent(8,3,6,rng)
        observations=rng.normal(size=(16,8)).astype(np.float32);actions=[];logp=[];values=[]
        for observation in observations:
            action,p,value=agent.act(observation,rng);actions.append(action);logp.append(p);values.append(value)
        advantages,returns=compute_gae(np.ones(16),np.zeros(16),values,0.,.99,.95)
        batch=dict(observations=observations,actions=np.asarray(actions),log_probabilities=np.asarray(logp),
            values=np.asarray(values),advantages=advantages,returns=returns)
        cfg=dict(epochs=2,minibatch_size=8,clip_ratio=.2,entropy_coefficient=.01,max_grad_norm=.5)
        metrics=agent.update(batch,cfg,rng);self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        metadata=dict(observation_size=8,action_size=3,hidden_size=6,actor_learning_rate=3e-4,
            critic_learning_rate=1e-3,repeat_logit=1.5,global_step=16)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.npz';agent.save(path,metadata);loaded,loaded_metadata=PPOAgent.load(path,np.random.default_rng(1))
            self.assertEqual(loaded_metadata['global_step'],16)
            self.assertTrue(np.allclose(agent.policy(observations)[0],loaded.policy(observations)[0]))


if __name__=='__main__':unittest.main()
