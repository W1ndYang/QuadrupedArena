import io
import json
from pathlib import Path
import secrets
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

from qwen_client import APIError,call_api,endpoint,load_key,parse_stream,save_key,usage_total
from qwen_trial import game_config,parse_action,probe,request_context,run,settings,system_message
from llm_trial import Match
from llm_report import audit


def stream(*chunks):
    return [b'data: '+json.dumps(c).encode()+b'\n' if c!='DONE' else b'data: [DONE]\n' for c in chunks]


class ClientTests(unittest.TestCase):
    def test_none_choice_reaches_wire(self):
        from unittest.mock import MagicMock
        opener=MagicMock();opener.open.return_value.__enter__.return_value=stream({'choices':[{'delta':{'content':'{}'},'finish_reason':'stop'}]},'DONE')
        with patch('qwen_client.urllib.request.build_opener',return_value=opener):
            call_api(dict(settings(),tool_choice='none'),'synthetic-test-secret',[],[{'type':'function'}])
        payload=json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(payload['tool_choice'],'none');self.assertTrue(payload['tools'])

    def test_official_beijing_endpoint(self):
        self.assertEqual(endpoint(settings()),'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')

    def test_untrusted_and_credential_urls_rejected(self):
        for url in ('https://dashscope.aliyuncs.com.evil.test/compatible-mode/v1','http://dashscope.aliyuncs.com/compatible-mode/v1','https://key@dashscope.aliyuncs.com/compatible-mode/v1','https://dashscope.aliyuncs.com/compatible-mode/v1?key=test'):
            with self.assertRaises(ValueError):endpoint(dict(base_url=url))

    def test_key_permissions_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'private/key';key=secrets.token_hex(16);save_key(key,p)
            self.assertEqual(p.stat().st_mode&0o777,0o600);self.assertEqual(load_key(p),key)
            with self.assertRaises(FileExistsError):save_key(secrets.token_hex(16),p)
            p.chmod(0o644)
            with self.assertRaises(ValueError):load_key(p)

    def test_key_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'key';p.symlink_to(Path(temp)/'another')
            with self.assertRaises(ValueError):save_key(secrets.token_hex(16),p)

    def test_stream_usage_and_reasoning_not_saved(self):
        chunks=stream({'id':'test','model':'mock-not-qwen','choices':[{'delta':{'reasoning_content':'private thinking'}}]},
            {'choices':[{'delta':{'content':'{"ok":true}'},'finish_reason':'stop'}]},
            {'choices':[],'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}},'DONE')
        result=parse_stream(chunks,0,clock=lambda:1)
        self.assertEqual(result['content'],'{"ok":true}');self.assertEqual(result['usage']['total_tokens'],30)
        self.assertNotIn('private thinking',json.dumps(result));self.assertEqual(result['reasoning_characters_received'],16)

    def test_incomplete_stream_rejected(self):
        with self.assertRaises(APIError):parse_stream(stream({'choices':[{'delta':{'content':'{}'},'finish_reason':'stop'}]}),0)

    def test_stream_tool_arguments_accumulate(self):
        result=parse_stream(stream({'choices':[{'delta':{'tool_calls':[{'index':0,'id':'call-1','function':{'name':'probe_geometry','arguments':'{"points":'}}]}}]},
            {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'[[0,0]]}'}}]},'finish_reason':'tool_calls'}]},'DONE'),0)
        self.assertEqual(json.loads(result['tool_calls'][0]['function']['arguments']),{'points':[[0,0]]})

    def test_usage_missing_is_not_zero(self):
        self.assertIsNone(usage_total([{'usage':None}])['total_tokens'])
        self.assertEqual(usage_total([{'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}}])['total_tokens'],0)

    def test_reasoning_tokens_not_added_twice(self):
        u=usage_total([{'usage':{'prompt_tokens':10,'completion_tokens':8,'total_tokens':18,'completion_tokens_details':{'reasoning_tokens':5}}}]*2)
        self.assertEqual(u['total_tokens'],36);self.assertEqual(u['reasoning_tokens'],10)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.config=settings();self.cfg=game_config(self.config);self.cfg.update(seed=12345,duration=4,lambda_max=0)
        self.match=Match(self.cfg)

    def request(self,index=0):
        return dict(observation_id=index,action=dict(skill='HOLD'),evidence='synthetic fixture',choice_summary='hold',alternatives_summary='none',expected_result='test only')

    def response(self,index=0):
        return dict(content=json.dumps(self.request(index)),tool_calls=[],usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15),
            response_model='mock-not-qwen',response_id='test',finish_reason='stop',api_wall_seconds=.01)

    def test_strict_json_and_action_validation(self):
        for content in ('```json\n{}\n```','{}',json.dumps(self.request(3)),json.dumps(dict(self.request(),action={'skill':'EXEC'}))):
            with self.assertRaises(ValueError):parse_action(content,self.match)
        self.assertEqual(self.match.t,0)

    def test_geometry_only_no_time_or_hp_changes(self):
        before=self.match.observe();result=probe(self.match,[[-4,-1.5]])
        self.assertTrue(result[0]['reachable']);self.assertEqual(before,self.match.observe())

    def test_prompt_does_not_include_random_seed_or_previous_trial_actions(self):
        text=system_message(self.cfg)['content'];self.assertNotIn('"seed"',text);self.assertNotIn('12.0 分',text)
        for phrase in ('项目背景','B 并不全知','工具调用本身不是运动命令','path_length=0','唯一成绩'):
            self.assertIn(phrase,text)

    def test_budget_reminder_is_per_call_without_mutating_history(self):
        messages=[system_message(self.cfg),dict(role='user',content='fixture')]
        original=json.dumps(messages)
        for attempt,choice in enumerate(('auto','auto','none')):
            current,config=request_context(messages,7,attempt,self.config)
            self.assertEqual(config['tool_choice'],choice)
            self.assertIn(f'剩余工具轮数 {2-attempt}',current[0]['content'])
            self.assertIn('observation_id=7',current[0]['content'])
        self.assertEqual(json.dumps(messages),original)

    def test_two_tool_rounds_then_forced_final(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'run';choices=[]
            def fake(config,key,messages,tools):
                choices.append(config['tool_choice'])
                if len(choices)<=2:
                    r=self.response();r.update(content='',finish_reason='tool_calls',tool_calls=[dict(id='c'+str(len(choices)),type='function',function=dict(name='probe_geometry',arguments='{"points":[[-4,-1.5]]}'))]);return r
                return self.response(0 if len(choices)==3 else 1)
            with patch('qwen_trial.game_config',return_value=self.cfg),redirect_stdout(io.StringIO()):run(self.config,directory,secrets.token_hex(16),client=fake)
            self.assertEqual(choices,['auto','auto','none','auto'])
            saved=json.loads((directory/'api_calls/000_02_request.json').read_text())
            self.assertEqual(saved['tool_choice'],'none')
            self.assertEqual(audit(directory)[0]['token_usage'],60)

    def test_mock_api_commits_and_usage_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'run';secret=secrets.token_hex(16);calls=[]
            def fake(config,key,messages,tools):
                index=sum(m['role']=='user' for m in messages)-1;calls.append(index);return self.response(index)
            with patch('qwen_trial.game_config',return_value=self.cfg),redirect_stdout(io.StringIO()):run(self.config,directory,secret,client=fake)
            result,*_=audit(directory)
            self.assertEqual(calls,[0,1]);self.assertEqual(result['token_usage'],30);self.assertEqual(result['simulation_seconds'],4)
            self.assertEqual(json.loads((directory/'validation.json').read_text())['token_usage'],30)
            for p in directory.rglob('*.json'):self.assertNotIn(secret,p.read_text())

    def test_invalid_output_stops_without_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'run';response=self.response();response['content']='not JSON'
            with patch('qwen_trial.game_config',return_value=self.cfg),redirect_stdout(io.StringIO()),self.assertRaises(APIError):
                run(self.config,directory,secrets.token_hex(16),client=lambda *a:response)
            status=json.loads((directory/'api_run_status.json').read_text());self.assertEqual(status['simulation_seconds'],0)
            self.assertEqual(status['api_calls'],1);self.assertEqual(list((directory/'decisions').glob('*.json')),[])

    def test_geometry_tool_then_final_action(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'run';n=0
            def fake(*args):
                nonlocal n
                n+=1
                if n==1:
                    r=self.response();r.update(content='',finish_reason='tool_calls',tool_calls=[dict(id='c1',type='function',function=dict(name='probe_geometry',arguments='{"points":[[-4,-1.5]]}'))]);return r
                return self.response(n-2)
            with patch('qwen_trial.game_config',return_value=self.cfg),redirect_stdout(io.StringIO()):run(self.config,directory,secrets.token_hex(16),client=fake)
            entries=json.loads((directory/'decisions.json').read_text());self.assertEqual(entries[0]['api_call_count'],2)
            self.assertEqual(entries[0]['tokens'],30);self.assertEqual(entries[0]['simulation_start'],0)


if __name__=='__main__':unittest.main()
