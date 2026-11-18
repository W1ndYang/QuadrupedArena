"""Protocol, public identity, and formal-counterfactual boundary tests."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from formal_central_screen import CentralController, run as formal_run
from llm_decision_core import EpisodeContext, _check_public_request
from llm_decision_service import ServiceState, fixture_response
from llm_fast_optimizer import _templates
from llm_motion_client import messages_for, APIReasoner, deepseek_client, qwen_client
from llm_map_visual import render as render_map_review
from llm_replay import _effective_config, replay
from llm_semantic import build_map, public_rules, validate_motion
from llm_trajectory import generate


def public_request(episode='test',request='test:1',time=0.,stage=0,peek=0,
                   observations=None):
    return dict(episode_id=episode,request_id=request,time=time,stage=stage,
        decision_type='initial' if peek==0 else 'observation_complete',
        public=dict(own_xy=[-8.,0.] if stage==0 else [3.,0.],
            own_yaw=0.,own_speed=0.,smoke_available=False,
            remaining_seconds=48.-time,
            action_mask=[False,False,True,True,True,True,True,False,False,False],
            peek_count=peek,stage_elapsed_seconds=time,
            last_action=None,skill_feedback='initial'),
        observations=observations or [],
        skill_state=dict(current_option=None,plan_request_id=None,
                         waits_done=0,recovery=False))


def sample(t,xy,stage=0):
    return dict(time=t,xy=list(xy),observer_xy=[-8.,-2.1] if stage==0
                else [3.,-2.1],observer_yaw=0.,uncertainty_m=0.,
                stage=stage,frame_valid=True)


class LLMDecisionTests(unittest.TestCase):
    def test_map_is_geometry_and_motion_protocol_rejects_invented_routes(self):
        semantic_map=build_map()
        self.assertEqual(semantic_map['map_id'],build_map()['map_id'])
        self.assertEqual(semantic_map['review']['status'],
                         'geometry_verified_manual_semantics_pending')
        track=dict(id='track-0001',region='first',
                   observations=[sample(3.1,[-3.,-.2]),sample(3.2,[-2.99,-.1])],
                   estimated_velocity=[.1,1.])
        decision=dict(schema='guard-motion-hypotheses-v1',request_id='req',
            map_id=semantic_map['map_id'],track_id=track['id'],
            region='first',game_time=3.3,horizon_seconds=8.,
            recent_observations=track['observations'],
            estimated_guard_velocity=track['estimated_velocity'])
        payload=json.loads(messages_for(decision,semantic_map,public_rules())[-1]['content'])
        self.assertEqual(payload['response_constraints']['allowed_next_target_ids'],
                         semantic_map['regions']['first']['node_ids']+['first'])
        response=fixture_response(payload)
        validate_motion(response,semantic_map,track['id'],'first',8.)
        trajectories=generate(semantic_map,response,track,3.3,8.,public_rules())
        self.assertEqual(len(trajectories),3)
        self.assertTrue(all(t['covered_until']>=8. for t in trajectories))
        broken=copy.deepcopy(response)
        broken['hypotheses'][0]['edge_ids'][0]='private_patrol_waypoint'
        with self.assertRaisesRegex(ValueError,'Unknown or non-guard edge'):
            validate_motion(broken,semantic_map,track['id'],'first',8.)

    def test_track_association_uses_global_time_and_region(self):
        context=EpisodeContext('e','map','rules')
        context.add_samples([sample(1.,[-3.,-1.]),sample(1.1,[-3.,-.9])])
        first=context.track_for(0)
        self.assertIsNotNone(first)
        context.add_samples([sample(20.,[7.8,1.],1),sample(20.1,[7.8,1.1],1)])
        second=context.track_for(1)
        self.assertNotEqual(first['id'],second['id'])
        self.assertEqual(len(first['observations']),2)
        self.assertEqual(len(second['observations']),2)
        context.add_samples([sample(21.,[-3.,-.5])])
        self.assertEqual(context.track_for(0)['id'],first['id'])

    def test_hidden_request_rejected_and_duplicate_id_not_reexecuted(self):
        invalid=public_request()
        invalid['private_patrol']={'waypoints':[]}
        with self.assertRaisesRegex(ValueError,'Unexpected decision request fields'):
            _check_public_request(invalid)
        with tempfile.TemporaryDirectory() as directory:
            state=ServiceState(dict(provider='deepseek'),directory,fixture=True)
            request=public_request()
            one=state.decide(request)
            two=state.decide(request)
            self.assertEqual(one,two)
            self.assertEqual(one['status'],'observe')
            self.assertEqual(len(state.orchestrator.episodes['test'].requests),1)
            changed=copy.deepcopy(request)
            changed['public']['own_xy']=[-7.,0.]
            result=state.decide(changed)
            self.assertEqual(result['status'],'error')
            self.assertIn('different public payload',result['error'])
            self.assertEqual(state.decide(request),one)

    def test_formal_hypothesis_rollout_ignores_episode_seed(self):
        hypothesis=dict(start_xy=[-3.,0.],start_yaw=1.57,
            speed_mps=1.,initial_speed=0.,
            waypoints=[[-3.,3.2],[-3.,-3.2]],
            dwell_node_indices=[0],dwell_seconds=.5)
        def episode(seed):
            return formal_run(seed,'north',0.,14.,4.,-3.9,4.,.6,True,
                second_peek=False,skip_peek=True,route_y=1.6,
                tag_seconds=.6,peek_if_unseen=False,stage2_local=False,
                patrol_style='hypothesis',hypothesis_guard=hypothesis)
        left,right=episode(0),episode(999999)
        self.assertEqual((left['reason'],left['seconds'],left['local_survived']),
                         (right['reason'],right['seconds'],right['local_survived']))

    def test_terminal_visual_batch_scores_prior_forecast_same_track(self):
        context=EpisodeContext('e','map','rules')
        context.add_samples([sample(1.,[-3.,0.]),sample(1.1,[-3.,.1])])
        track=context.track_for(0)
        context.predictions.append(dict(forecast_id='f1',track_id=track['id'],
            generated_at=1.1,horizon=1.,last_observed_xy=[-3.,.1],
            trajectories=[dict(hypothesis_id='north',probability=1.,
                covered_until=1.,states=[
                    dict(dt=round(i*.1,2),xy=[-3.,.1+i*.1],yaw=1.57,speed=1.)
                    for i in range(11)])]))
        context.add_samples([sample(1.5,[-3.,.5])])
        result=context.evaluation()
        self.assertEqual(result['samples'],1)
        self.assertLess(result['mean_position_error_m'],1e-6)
        self.assertTrue(result['coverage80'])
        self.assertEqual(sum(x['count'] for x in result['reliability_proxy']),1)

    def test_hypothesis_smoke_uses_online_public_target_rule(self):
        hypothesis=dict(start_xy=[-3.,0.],start_yaw=1.57,
            speed_mps=1.,initial_speed=0.,waypoints=[[-3.,3.2]],
            dwell_node_indices=[],dwell_seconds=0.)
        observed=[]
        original=CentralController.smoke_location_for
        def checked(controller,packet,target):
            observed.append(controller.smoke_uses_track)
            return original(controller,packet,target)
        with patch.object(CentralController,'smoke_location_for',checked):
            formal_run(0,'north',0.,14.,4.,-3.9,4.,.6,True,
                second_peek=False,skip_peek=True,route_y=1.6,
                tag_seconds=.6,peek_if_unseen=False,stage2_local=False,
                patrol_style='hypothesis',hypothesis_guard=hypothesis,
                use_smoke=True)
        self.assertTrue(observed)
        self.assertEqual(set(observed),{False})

    def test_recorded_replay_uses_service_configuration(self):
        events=[dict(kind='decision',config=dict(maximum_formal_particles=1)),
                dict(kind='decision',config=dict(maximum_formal_particles=1))]
        supplied=dict(maximum_formal_particles=4)
        self.assertEqual(_effective_config(events,supplied,False),
                         (events[0]['config'],'recorded_service'))
        self.assertEqual(_effective_config(events,supplied,True),
                         (supplied,'supplied_live_model'))
        events[1]['config']={'maximum_formal_particles':2}
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            _effective_config(events,supplied,False)

    def test_invalid_llm_target_is_logged_and_replayable(self):
        class InvalidTargetReasoner:
            def infer(self,messages):
                payload=json.loads(messages[-1]['content'])
                response=fixture_response(payload)
                response['hypotheses'][0]['next_target_id']='invented-waypoint'
                return dict(text=json.dumps(response),model='test-model',
                            usage={'output_tokens':123},wall_seconds=.01)

        with tempfile.TemporaryDirectory() as directory:
            state=ServiceState(dict(provider='deepseek'),directory,fixture=True)
            state.orchestrator.reasoner=InvalidTargetReasoner()
            request=public_request(time=3.3,peek=1,observations=[
                sample(3.1,[-3.,0.]),sample(3.2,[-3.,.1])])
            result=state.decide(request)
            self.assertEqual(result['status'],'error')
            self.assertIn('invented-waypoint',result['error'])
            event=json.loads(state.log_path.read_text().splitlines()[0])
            self.assertEqual(event['llm_response']['usage']['output_tokens'],123)
            self.assertIn('invented-waypoint',event['llm_response']['text'])
            output=Path(directory)/'replay.json'
            report=replay(state.log_path,output,state.config)
            self.assertEqual((report['count'],report['same_decisions'],
                              report['replayable_count']),(1,1,1))
            self.assertEqual(report['rows'][0]['replayed_status'],'error')
            self.assertEqual(json.loads(output.read_text())['same_decisions'],1)

            event.pop('llm_response')
            old_log=Path(directory)/'old_events.jsonl'
            old_log.write_text(json.dumps(event)+'\n')
            old_report=replay(old_log,Path(directory)/'old_replay.json',state.config)
            self.assertEqual(old_report['unreplayable_request_ids'],['test:1'])
            self.assertEqual(old_report['same_decisions'],0)

    def test_fast_proxy_uses_pregenerated_formal_robot_routes(self):
        templates=_templates()
        self.assertEqual(len(templates['rows']),24)
        self.assertEqual(len({(row['stage'],row['side'],row['wait'],row['smoke'])
                              for row in templates['rows']}),24)
        with tempfile.TemporaryDirectory() as directory:
            state=ServiceState(dict(provider='deepseek',optimizer_mode='fast_proxy',
                                    maximum_formal_particles=4),directory,fixture=True)
            self.assertEqual(templates['map_id'],state.orchestrator.map['map_id'])
            request=public_request(time=3.3,peek=1,observations=[
                sample(3.1,[-3.,0.]),sample(3.2,[-3.,.1])])
            result=state.decide(request)
            self.assertEqual(result['status'],'plan')
            self.assertEqual(result['optimizer']['scoring_method'],
                             'fast_los_proxy_v1')
            self.assertEqual(len(result['optimizer']['candidates']),6)
            self.assertLess(result['timing']['optimizer_seconds'],1.)

    def test_map_review_renders_readable_artifacts_without_approval(self):
        semantic_map=build_map()
        with tempfile.TemporaryDirectory() as directory:
            outputs=render_map_review(semantic_map,directory)
            self.assertEqual(semantic_map['review']['status'],
                             'geometry_verified_manual_semantics_pending')
            self.assertTrue(all(Path(path).stat().st_size>0
                                for path in outputs.values()))
            self.assertGreater(Path(outputs['png']).stat().st_size,1000)
            notes=Path(outputs['checklist']).read_text()
            self.assertIn('Guard 1 region spans x=-3.9..-2.1',notes)
            self.assertIn('actual guard waypoints and order remain unknown',notes)

    def test_live_service_keeps_pending_semantics_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            config=json.loads((Path(__file__).parent/
                'config/llm_decision_deepseek.json').read_text())
            state=ServiceState(config,directory,fixture=False)
            self.assertEqual(state.map_review_status,
                             'geometry_verified_manual_semantics_pending')
            response=state.decide(public_request())
            self.assertEqual(response['status'],'observe')
            event=json.loads(state.log_path.read_text().splitlines()[0])
            self.assertEqual(event['map_review_status'],state.map_review_status)

    def test_service_import_order_uses_current_api_adapters(self):
        # A fresh process matters: fixture tests and earlier imports can hide
        # the frozen simulator's same-name modules in sys.modules.
        code="""
import json, sys
from pathlib import Path
import llm_decision_service
import llm_motion_client as client
root=Path.cwd().parents[1]
assert Path(sys.modules['deepseek_client'].__file__).resolve()!=root/'scripts/deepseek_client.py'
assert Path(client.deepseek_client.__file__).resolve()==root/'scripts/deepseek_client.py'
assert Path(client.qwen_client.__file__).resolve()==root/'scripts/qwen_client.py'
assert client.deepseek_client.parse_stream is client.qwen_client.parse_stream
config=json.loads(Path('config/llm_decision_deepseek.json').read_text())
client.APIReasoner(config)
assert client.deepseek_client.payload(config,[])['thinking']=={'type':'disabled'}
"""
        subprocess.run([sys.executable,'-c',code],
                       cwd=Path(__file__).resolve().parent,check=True,
                       capture_output=True,text=True)

    def test_live_reasoner_reaches_current_adapter_with_thinking_disabled(self):
        config=json.loads((Path(__file__).parent/
            'config/llm_decision_deepseek.json').read_text())
        reasoner=APIReasoner(config)
        def fake_api(actual_config,key,messages):
            self.assertEqual(deepseek_client.payload(actual_config,messages)
                             ['thinking'],{'type':'disabled'})
            return dict(content='{}',api_wall_seconds=.25,
                        time_to_first_output_seconds=.1,
                        time_to_first_visible_content_seconds=.1,
                        usage={'completion_tokens':2})
        with patch.dict('os.environ',DEEPSEEK_API_KEY='fake-unit-test-key'), \
                patch.object(deepseek_client,'call_api',side_effect=fake_api) as call:
            response=reasoner.infer([dict(role='user',content='test')])
        self.assertEqual(call.call_count,1)
        self.assertEqual(response['wall_seconds'],.25)
        self.assertEqual(response['raw_metadata']['time_to_first_output_seconds'],.1)


if __name__=='__main__':unittest.main()
