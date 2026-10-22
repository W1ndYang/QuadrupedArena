"""Offline tests only: no credentials, paid APIs, or GPU required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from adaptive_motion import load_config
from deepseek_client import APIError,NoRedirect,endpoint,payload,parse_stream,save_key,read_private_key,call_api
from deepseek_trial import PROJECT,Planner,observation_message,public_observation,settings,system_message,motion_request,validate_model_output
from semantic_skills import SemanticSkills
from vision_core import VisionMatch,public_packet


class DeepSeekTests(unittest.TestCase):
    def setUp(self):
        self.config=dict(settings(),explain_decisions=False,decision_fields=['observation_id','skill','params']);self.cfg=load_config(PROJECT/'configs/trial_rotating.json')
        self.engine=SemanticSkills(self.cfg);self.match=VisionMatch(self.cfg)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/'frames').mkdir();Image.new('RGB',(768,432),'gray').save(self.root/'frames/000.png')
        frame=dict(file='frames/000.png',simulation_seconds=0,own_xy=self.cfg['start_a'],body_yaw_degrees=0,
            camera_pan_degrees=0,camera_world_yaw_degrees=0,camera_eye=[-4.64,-2.8,.6],horizontal_fov_degrees=90)
        self.packet=public_packet(0,self.match.states[-1],[frame],self.cfg['start_b']);self.packet['skills']=self.engine.public(0)
        self.action=dict(observation_id=0,skill='OBSERVE',params=dict(direction='north'))
    def response(self,action=None):
        return dict(content=json.dumps(action or self.action),tool_calls=[],finish_reason='stop',response_model='deepseek-flash',response_id='test',
            usage=dict(prompt_tokens=100,completion_tokens=20,total_tokens=120,completion_tokens_details=dict(reasoning_tokens=8)),
            api_wall_seconds=.3,api_started_utc='2026-10-21T12:00:00+00:00',api_finished_utc='2026-10-21T12:00:00.300000+00:00',time_to_first_output_seconds=.1,time_to_first_visible_content_seconds=.2)
    def test_official_endpoint_only(self):
        for url in ('http://api.deepseek.com','https://api.deepseek.com.evil.test','https://user@api.deepseek.com','https://api.deepseek.com/beta','https://api.deepseek.com?token=x'):
            with self.assertRaises(ValueError):endpoint(dict(self.config,base_url=url))
    def test_configured_thinking_payload(self):
        for effort in ('low','high','max'):
            self.assertEqual(payload(dict(self.config,reasoning_effort=effort),[])['reasoning_effort'],effort)
        p=payload(self.config,[]);self.assertEqual(p['thinking'],{'type':'enabled'});self.assertEqual(p['reasoning_effort'],self.config['reasoning_effort'])
        self.assertEqual(p['response_format'],{'type':'json_object'});self.assertNotIn('enable_thinking',p);self.assertNotIn('temperature',p);self.assertNotIn('tools',p)
        with self.assertRaises(ValueError):payload(dict(self.config,reasoning_effort='invalid'),[])
    def test_redirect_blocked(self):
        with self.assertRaises(APIError):NoRedirect().redirect_request(None,None,302,'',{},'https://evil.test')
    def test_maximum_platform_limit_profile(self):
        cfg=settings(PROJECT/'configs/deepseek_max_platform_limit.json')
        self.assertEqual(payload(cfg,[])['max_tokens'],393216)
        self.assertEqual(cfg['reasoning_effort'],'max')
        self.assertIsNone(cfg['wall_timeout_seconds']);self.assertIsNone(cfg['launch_timeout_seconds'])
        self.assertEqual(cfg['timeout_seconds'],600)
        self.assertEqual(system_message(self.cfg,cfg),system_message(self.cfg,dict(settings(),reasoning_effort='max')))
        self.assertEqual(settings()['reasoning_effort'],'low')
    def test_optional_token_limit_and_validation(self):
        for value in (None,):self.assertNotIn('max_tokens',payload(dict(self.config,max_tokens=value),[]))
        for value in (0,-1,393217,True,1.5):
            with self.assertRaises(ValueError):payload(dict(self.config,max_tokens=value),[])
    def test_profile_override_environment(self):
        with patch.dict('os.environ',ARENA_DEEPSEEK_CONFIG=str(PROJECT/'configs/deepseek_max_platform_limit.json')):
            self.assertEqual(settings()['max_tokens'],393216)
    def mock_long_stream(self,config):
        now=[0.]
        chunk=dict(model='deepseek-flash',choices=[dict(index=0,delta=dict(content='{"ok":true}'),finish_reason='stop')],
            usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15))
        class Response:
            def __enter__(self):now[0]=1000.;return self
            def __exit__(self,*args):return False
            def __iter__(self):return iter([b'data: '+json.dumps(chunk).encode(),b'data: [DONE]'])
        with patch('deepseek_client.urllib.request.build_opener') as opener,patch('deepseek_client.time.perf_counter',side_effect=lambda:now[0]):
            opener.return_value.open.return_value=Response()
            response=call_api(config,'fake-key-not-written',[])
            self.assertEqual(opener.return_value.open.call_args.kwargs['timeout'],config['timeout_seconds'])
            return response
    def test_disabled_wall_deadline_allows_long_stream(self):
        cfg=settings(PROJECT/'configs/deepseek_max_platform_limit.json')
        r=self.mock_long_stream(cfg)
        self.assertEqual(r['api_wall_seconds'],1000.);self.assertEqual(r['finish_reason'],'stop')
    def test_legacy_wall_deadline_remains_enforced(self):
        with self.assertRaises(APIError):self.mock_long_stream(self.config)
    def test_private_key_storage(self):
        p=self.root/'credentials/deepseek.key';save_key('test-not-a-real-api-key',p)
        self.assertEqual(p.stat().st_mode&0o777,0o600);self.assertEqual(read_private_key(p),'test-not-a-real-api-key')
        with self.assertRaises(FileExistsError):save_key('test-not-a-real-api-key',p)
    def test_public_rules_no_policy(self):
        c=copy.deepcopy(self.cfg);c['policy']['hidden_sentinel']='PRIVATE';c['seed']='PRIVATE'
        text=system_message(c)['content'];self.assertNotIn('PRIVATE',text);self.assertIn('hold_seconds',text);self.assertIn('tau',text)
    def test_observation_no_extra_fields(self):
        p=copy.deepcopy(self.packet);p['enemy_xy']=[1,2]
        with self.assertRaises(ValueError):public_observation(p)
        p=copy.deepcopy(self.packet);p['self_state']['enemy_hp']=3
        with self.assertRaises(ValueError):public_observation(p)
    def test_image_and_bounded_history(self):
        message,manifest=observation_message(self.packet,self.root,self.config,list(range(10)))
        self.assertEqual(json.loads(message['content'][0]['text'])['recent_decisions'],[6,7,8,9])
        self.assertTrue(message['content'][2]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertEqual(message['content'][2]['image_url']['detail'],'original');self.assertEqual(len(manifest),1)
    def test_future_or_outside_image_rejected(self):
        p=copy.deepcopy(self.packet);p['frames'][0]['simulation_seconds']=1
        with self.assertRaises(ValueError):observation_message(p,self.root,self.config,[])
        p=copy.deepcopy(self.packet);p['frames'][0]['file']='../private.png'
        with self.assertRaises(ValueError):observation_message(p,self.root,self.config,[])
    def test_successful_mock_decision_compile(self):
        planner=Planner(self.config,self.cfg,'fake-key-not-written',self.root,client=lambda *a:self.response())
        request,timing=planner.decide(self.packet,self.engine.validate)
        a,route,cam,status=self.engine.commit(request,self.packet)
        self.assertEqual(request,self.action);self.assertAlmostEqual(a['facing'],1.5707963267948966)
        self.assertEqual(timing['tokens'],120);self.assertEqual(timing['reasoning_tokens'],8)
        self.assertEqual(timing['reasoning_effort'],self.config['reasoning_effort'])
        self.assertEqual(timing['api_started_utc'],'2026-10-21T12:00:00+00:00')
        self.assertNotIn('reason',request)
        for p in self.root.rglob('*.json'):self.assertNotIn('fake-key-not-written',p.read_text())
        with self.assertRaises(ValueError):planner.decide(self.packet,self.engine.validate)
    def test_invalid_action_not_repaired_or_retried(self):
        response=self.response(dict(self.action,skill='TELEPORT'));calls=[]
        def client(*a):calls.append(1);return response
        planner=Planner(self.config,self.cfg,'fake',self.root,client=client)
        with self.assertRaises(ValueError):planner.decide(self.packet,self.engine.validate)
        self.assertEqual(len(calls),1);self.assertFalse((self.root/'actions').exists());self.assertEqual(planner.history,[])
    def test_incomplete_response_rejected(self):
        r=self.response();r['finish_reason']='length'
        planner=Planner(self.config,self.cfg,'fake',self.root,client=lambda *a:r)
        with self.assertRaises(ValueError):planner.decide(self.packet,self.engine.validate)
    def test_explanation_field_rejected(self):
        planner=Planner(self.config,self.cfg,'fake',self.root,client=lambda *a:self.response(dict(self.action,reason='不需要的解释')))
        with self.assertRaises(ValueError):planner.decide(self.packet,self.engine.validate)
    def assessed_action(self):
        return dict(self.action,assessment=dict(visual_evidence=[dict(time=0,opponent_visible=None,observation='灰色测试图，无法确认目标。')],
            state_evidence='自身3HP。',decision_summary='观察北面。',uncertainty='当前图像没有可辨认地标。'))
    def test_assessment_does_not_change_motion(self):
        cfg=dict(self.config,explain_decisions=True);r=self.assessed_action()
        planner=Planner(cfg,self.cfg,'fake',self.root,client=lambda *a:self.response(r))
        request,timing=planner.decide(self.packet,self.engine.validate)
        self.assertEqual(request,r);self.assertEqual(planner.history,[self.action]);self.assertEqual(motion_request(request),self.action)
        self.engine.commit(motion_request(request),self.packet)
    def test_assessment_requires_current_frames(self):
        cfg=dict(self.config,explain_decisions=True);r=self.assessed_action();r['assessment']['visual_evidence'][0]['time']=2
        with self.assertRaises(ValueError):validate_model_output(r,self.packet,cfg,self.engine.validate)
        r=self.assessed_action();r['assessment']['visual_evidence']=[]
        with self.assertRaises(ValueError):validate_model_output(r,self.packet,cfg,self.engine.validate)
    def test_assessment_prompt_has_no_conflicting_output_requirement(self):
        text=system_message(self.cfg,dict(self.config,explain_decisions=True))['content']
        self.assertIn('assessment',text);self.assertNotIn('不要输出reason、理由',text);self.assertNotIn('三字段',text)
    def test_failure_external_timestamp(self):
        def fail(*args):raise APIError('test failure')
        planner=Planner(self.config,self.cfg,'fake',self.root,client=fail)
        with self.assertRaises(APIError):planner.decide(self.packet,self.engine.validate)
        record=json.loads((self.root/'api_calls/000_failure.json').read_text())
        self.assertIn('call_start_utc',record);self.assertIn('call_failed_utc',record);self.assertGreater(record['call_until_failure_seconds'],0)
    def test_request_budget(self):
        planner=Planner(self.config,self.cfg,'fake',self.root,client=lambda *a:self.fail('must not call'))
        planner.calls=20
        with self.assertRaises(ValueError):planner.decide(self.packet,self.engine.validate)
    def test_sse_usage_and_reasoning_not_saved(self):
        chunks=[dict(choices=[dict(index=0,delta=dict(reasoning_content='private reasoning'))]),
            dict(model='deepseek-flash',choices=[dict(index=0,delta=dict(content='{"ok":true}'),finish_reason='stop')]),
            dict(choices=[],usage=dict(prompt_tokens=11,completion_tokens=4,total_tokens=15))]
        lines=[b'data: '+json.dumps(c).encode() for c in chunks]+[b'data: [DONE]']
        r=parse_stream(lines,0,clock=lambda:1)
        self.assertEqual(r['usage']['total_tokens'],15);self.assertNotIn('private reasoning',json.dumps(r));self.assertFalse(r['reasoning_text_saved'])


if __name__=='__main__':unittest.main()
