"""Time, motion and Option-boundary contracts against the actual executor."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_decision_core import DecisionOrchestrator,_recent_velocity,_velocity
from llm_decision_service import ServiceState,fixture_response
from llm_motion_client import FixtureReasoner,messages_for
from llm_semantic import build_map,public_rules
from llm_trajectory import generate,interpolate,local_motion_envelope
from llm_robot_rollout import rollout,candidates,prewarm,_candidates
from llm_option_policy import LLMOptionPolicy
from llm_replay import replay,evaluation_matches
from option_env import run_episode
from test_llm_decision import sample,public_request


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.map=build_map();self.rules=public_rules()

    def test_replay_tolerates_roundoff_but_keeps_counts_and_outcomes_exact(self):
        left=dict(samples=13,brier=.7733333333333328,failed=True)
        self.assertTrue(evaluation_matches(left,dict(left,brier=.7733333333333325)))
        self.assertFalse(evaluation_matches(left,dict(left,brier=.7733)))
        self.assertFalse(evaluation_matches(left,dict(left,samples=14)))
        self.assertFalse(evaluation_matches(left,dict(left,failed=False)))

    def test_numeric_stationary_gate_skips_api_and_candidate_rollouts(self):
        class ForbiddenReasoner:
            def infer(self,messages):raise AssertionError('Numeric mode must not call API')
        model=DecisionOrchestrator(ForbiddenReasoner(),dict(max_llm_calls_per_stage=0,
            max_peeks_per_stage=2,robot_rollout_mode='stateful_v2'))
        request=public_request(time=3.3,peek=1,observations=[
            sample(3.1,[-3.,0.]),sample(3.2,[-3.,0.])])
        with patch('llm_decision_core.prediction_horizon',
                   side_effect=AssertionError('Information gate should precede rollout')):
            response=model.decide(request)
        self.assertEqual(response['status'],'observe')
        self.assertFalse(response['llm_called'])
        self.assertEqual(response['timing']['preparation_seconds'],0.)

    def motion(self,xy,velocity,state='MOVE',dwell=(0.,0.)):
        track=dict(id='track-0001',region='first',estimated_velocity=list(velocity),
                   observations=[sample(1.,xy)])
        decision=dict(schema='guard-motion-hypotheses-v1',request_id='r',
                      map_id=self.map['map_id'],track_id=track['id'],region='first',
                      recent_observations=track['observations'],estimated_guard_velocity=list(velocity))
        payload=json.loads(messages_for(decision,self.map,self.rules)[-1]['content'])
        response=fixture_response(payload)
        response['hypotheses'][0]['state']=state
        response['hypotheses'][0]['dwell_range_seconds']=list(dwell)
        return track,response

    def test_guard_does_not_return_to_graph_start_and_honors_initial_dwell(self):
        track,response=self.motion([-3.28,3.08],[0.,0.])
        trace=generate(self.map,response,track,1.,3.,self.rules,
                       mode='observed_anchor_v2')[0]
        self.assertLess(interpolate(trace['states'],1.)['xy'][1],3.)
        self.assertAlmostEqual(trace['states'][0]['xy'][0],-3.28)
        track,response=self.motion([-3.28,3.08],[0.,0.],state='DWELL',dwell=(1.,1.))
        trace=generate(self.map,response,track,1.4,2.,self.rules,
                       mode='observed_anchor_v2')[0]
        self.assertLess(math.dist(trace['states'][0]['xy'],[-3.28,3.08]),1e-7)
        self.assertAlmostEqual(trace['initial_dwell_remaining'],.6)
        self.assertEqual(trace['source_observed_at'],1.)
        self.assertAlmostEqual(trace['source_age_seconds'],.4)

    def test_time_advancement_matches_one_continuous_forecast(self):
        track,response=self.motion([-3.,2.7],[0.,-1.])
        initial=generate(self.map,response,track,1.,5.,self.rules,
                         mode='observed_anchor_v2')
        for elapsed in (0.,1.,3.):
            later=generate(self.map,response,track,1.+elapsed,2.,self.rules,
                           mode='observed_anchor_v2')
            for left,right in zip(initial,later):
                self.assertLess(math.dist(interpolate(left['states'],elapsed)['xy'],
                                          right['states'][0]['xy']),1e-7)
                self.assertLess(math.dist(interpolate(left['states'],elapsed+1.)['xy'],
                                          interpolate(right['states'],1.)['xy']),1e-7)

    def test_boundary_reaction_history_requires_fresh_continuous_samples(self):
        track=dict(id='track-0001',region='first',estimated_velocity=[0.,-1.],
                   observations=[sample(1.,[-3.,0.]),sample(1.1,[-3.,-.1]),sample(1.2,[-3.,-.2])])
        fresh,_=local_motion_envelope(self.map,track,1.2,2.,self.rules)
        self.assertAlmostEqual(fresh[0]['possible_reaction_elapsed_seconds'],.2)
        stale,_=local_motion_envelope(self.map,track,1.7,2.,self.rules)
        self.assertEqual(stale[0]['possible_reaction_elapsed_seconds'],0.)
        track['observations']=[sample(.5,[-3.,.5]),sample(1.2,[-3.,-.2])]
        gapped,_=local_motion_envelope(self.map,track,1.2,2.,self.rules)
        self.assertEqual(gapped[0]['possible_reaction_elapsed_seconds'],0.)

    def test_recent_velocity_tracks_braking_and_preserves_noisy_fallback(self):
        rows=[sample(i*.1,[-3.,1.+i*.1-.9*(i*.1)**2]) for i in range(5)]
        recent=_recent_velocity(rows)
        self.assertAlmostEqual(recent[1],.28)
        self.assertGreater(abs(_velocity(rows)[1]-.28),.2)
        noisy=copy.deepcopy(rows)
        for row in noisy:row['uncertainty_m']=.2
        self.assertEqual(_recent_velocity(noisy),_velocity(noisy))
        stopped=rows+[sample(.5,rows[-1]['xy']),sample(.6,rows[-1]['xy'])]
        self.assertEqual(_recent_velocity(stopped),[0.,0.])

    def test_own_motion_matches_live_skill_from_peek_return_state(self):
        class Client:
            def __init__(self):self.requests=[]
            def decide(client,request):
                client.requests.append(copy.deepcopy(request))
                base=dict(episode_id=request['episode_id'],request_id=request['request_id'])
                if request['public']['peek_count']==0:
                    return dict(base,status='observe',action=2)
                if request['public']['action_mask'][7]:
                    return dict(base,status='boundary',action=7)
                return dict(base,status='plan',plan=dict(side='north',wait=0.,smoke=False),
                            boundary_recheck=True)
        class TracedPolicy(LLMOptionPolicy):
            def __init__(policy,*args):super().__init__(*args);policy.trace=[]
            def observe(policy,obs,mask):
                super().observe(obs,mask)
                policy.trace.append((policy.last_time,[float(obs[policy.index['self_x']])*12.,
                                                       float(obs[policy.index['self_y']])*8.]))
        client=Client();policy=TracedPolicy('motion-check',client,48.)
        summary,records=run_episode(810000,820000,policy,48.)
        self.assertEqual(summary['seconds'],7.48)
        request=client.requests[1];public=request['public']
        self.assertLess(public['own_speed'],-1.)
        self.assertAlmostEqual(public['own_yaw'],-math.pi/2)
        prewarm();before=_candidates.cache_info()
        predicted=next(row for row in candidates(public,0) if row['side']=='north'
                       and row['wait']==0. and not row['smoke'])
        self.assertEqual(_candidates.cache_info().hits,before.hits+1)
        self.assertTrue(predicted['complete'])
        self.assertEqual(predicted['end_boundary'],'second_cover')
        self.assertGreater(predicted['seconds'],7.)
        for dt in (1.,2.,3.,3.8,4.1):
            actual=min(policy.trace,key=lambda row:abs(row[0]-request['time']-dt))[1]
            expected=min(predicted['states'],key=lambda row:abs(row['dt']-dt))['xy']
            self.assertLess(math.dist(actual,expected),1e-4)
        boundary=client.requests[2]
        self.assertEqual(boundary['decision_type'],'gate_boundary')
        self.assertEqual(boundary['time'],6.5)
        self.assertTrue(boundary['observations'])
        self.assertTrue(all(row['time']>3.3 for row in boundary['observations']))
        self.assertEqual([row['action'] for row in records],[2,6,7])
        changed=dict(public,own_speed=public['own_speed']+.2)
        before=_candidates.cache_info()
        changed_rows=candidates(changed,0)
        self.assertEqual(_candidates.cache_info().misses,before.misses+1)
        self.assertAlmostEqual(changed_rows[0]['initial_state']['speed'],changed['own_speed'])

    def test_boundary_reassessment_calls_no_llm_and_replays(self):
        class ForbiddenReasoner:
            def infer(self,messages):raise AssertionError('Boundary must not call the API')
        config=dict(provider='deepseek',boundary_recheck=True,
                    robot_rollout_mode='stateful_v2',maximum_fast_particles=18)
        with tempfile.TemporaryDirectory() as directory:
            state=ServiceState(config,directory,fixture=True)
            state.orchestrator.reasoner=ForbiddenReasoner()
            request=public_request(time=6.5,peek=1,observations=[
                sample(6.4,[-2.517,-1.029]),sample(6.5,[-2.501,-1.116])])
            request['decision_type']='gate_boundary'
            request['public'].update(own_xy=[-5.302,1.54],own_yaw=.197,own_speed=2.2,
                last_action=6,action_mask=[False,True,False,False,True,False,False,True,False,False])
            result=state.decide(request)
            self.assertEqual(result['status'],'boundary')
            self.assertIn(result['action'],(1,4,7))
            self.assertFalse(result['llm_called'])
            self.assertEqual(result['timing']['api_seconds'],0.)
            self.assertEqual(result['forecast']['generated_at'],6.5)
            report=replay(state.log_path,Path(directory)/'replay.json',config)
            self.assertEqual(report['same_decisions'],1)

    def test_repeek_reuses_local_motion_after_stage_api_budget(self):
        calls=[]
        def inference(payload):
            calls.append(payload)
            if len(calls)>1:raise AssertionError('Exceeded one call per stage')
            return fixture_response(payload,compact=True)
        config=dict(provider='deepseek',optimizer_mode='fast_proxy',
            robot_rollout_mode='stateful_v2',trajectory_mode='observed_anchor_v2',
            maximum_fast_particles=18,observation_gate_mode='action_risk',
            max_llm_calls_per_stage=1,max_peeks_per_stage=2)
        model=DecisionOrchestrator(FixtureReasoner(inference),config)
        first=public_request(time=3.3,peek=1,observations=[
            sample(3.1,[-3.,0.]),sample(3.2,[-3.,.1])])
        model.decide(first)
        second=public_request(request='test:2',time=7.2,peek=2,observations=[
            sample(7.,[-3.,-2.]),sample(7.1,[-3.,-2.1])])
        response=model.decide(second)
        self.assertEqual(len(calls),1)
        self.assertFalse(response['llm_called'])
        self.assertEqual(response['timing']['api_seconds'],0.)
        self.assertEqual(response['forecast']['forecast_source'],'local_motion_envelope_v1')
        self.assertEqual(response['status'],'plan')

    def test_retreat_returns_to_cover_and_clears_gate_state(self):
        class Recovered(Exception):pass
        requests=[]
        class Client:
            def decide(self,request):
                requests.append(copy.deepcopy(request))
                base=dict(episode_id=request['episode_id'],request_id=request['request_id'])
                if request['skill_state']['recovery'] and request['public']['peek_count']==0:
                    raise Recovered()
                if request['public']['peek_count']==0:
                    return dict(base,status='observe',action=2)
                if request['public']['action_mask'][7]:
                    return dict(base,status='boundary',action=1)
                return dict(base,status='plan',plan=dict(side='north',wait=0.,smoke=False),
                            boundary_recheck=True)
        policy=LLMOptionPolicy('retreat-check',Client(),48.)
        with self.assertRaises(Recovered):
            run_episode(810000,820000,policy,48.)
        recovered=requests[-1]
        self.assertFalse(recovered['public']['action_mask'][7])
        self.assertTrue(recovered['public']['action_mask'][2])
        self.assertTrue(recovered['public']['action_mask'][3])
        self.assertEqual(recovered['skill_state']['retreats'],1)
        self.assertAlmostEqual(recovered['skill_state']['observation_window_elapsed'],0.)
        self.assertLess(math.dist(recovered['public']['own_xy'],[-8.,0.]),.15)


if __name__=='__main__':unittest.main()
