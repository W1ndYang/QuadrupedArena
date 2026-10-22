"""Offline only; never sends paid requests or reads credentials."""
import copy
import json
import unittest
from unittest.mock import MagicMock,patch
from qwen_vision import settings,vision_client
from deepseek_trial import system_message,parse_output
from adaptive_motion import load_config
from llm_trial import PROJECT


class QwenVisionTests(unittest.TestCase):
    def wire(self,mode,messages=None):
        opener=MagicMock()
        opener.open.return_value.__enter__.return_value=[b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}',b'data: [DONE]']
        with patch('qwen_client.urllib.request.build_opener',return_value=opener):
            result=vision_client(settings(mode),'synthetic-not-a-key',messages or [])
        return json.loads(opener.open.call_args.args[0].data),result
    def test_max_uses_provider_default_budget_without_fake_effort(self):
        p,r=self.wire('max');self.assertTrue(p['enable_thinking']);self.assertNotIn('thinking_budget',p)
        self.assertNotIn('reasoning_effort',p);self.assertEqual(p['model'],'qwen3.6-27b')
        self.assertIn('api_started_utc',r);self.assertIn('api_finished_utc',r)
    def test_min_really_disables_thinking(self):
        p,_=self.wire('min');self.assertFalse(p['enable_thinking']);self.assertNotIn('thinking_budget',p)
        self.assertNotIn('tools',p);self.assertEqual(p['max_tokens'],16384)
    def test_original_png_preserved_and_no_mutation(self):
        message=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='data:image/png;base64,TEST',detail='high'))])]
        original=copy.deepcopy(message);p,_=self.wire('max',message)
        self.assertEqual(message,original);self.assertNotIn('detail',p['messages'][0]['content'][0]['image_url'])
        self.assertEqual(p['messages'][0]['content'][0]['image_url']['url'],original[0]['content'][0]['image_url']['url'])
    def test_same_public_prompt_both_modes(self):
        cfg=load_config(PROJECT/'configs/trial_rotating.json')
        a=system_message(cfg,settings('max'));b=system_message(cfg,settings('min'))
        self.assertEqual(a,b);self.assertIn('assessment',a['content']);self.assertNotIn('class HumanPolicy',a['content'])
    def test_single_fence_only_no_json_repair(self):
        self.assertEqual(parse_output('```json\n{"skill":"OBSERVE"}\n```',True),{'skill':'OBSERVE'})
        for s in ('```json\n{}\n```','Here is JSON: {}','```json\n{"bad":}\n```','```json\n{}\n```\nprose'):
            with self.assertRaises(ValueError):parse_output(s,False)
        for s in ('Here is JSON: {}','```json\n{"bad":}\n```','```json\n{}\n```\nprose'):
            with self.assertRaises(ValueError):parse_output(s,True)


if __name__=='__main__':unittest.main()
