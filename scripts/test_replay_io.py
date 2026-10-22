import copy
import json
from pathlib import Path
import tempfile
import unittest

from adaptive_motion import load_config
from llm_trial import Match
from llm_report import audit
from replay_io import load_replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        cfg=load_config(Path(__file__).resolve().parents[1]/'configs/trial_rotating.json')
        cfg['duration']=2.;cfg['seed']=12345
        match=Match(cfg);obs=match.observe()
        request=dict(observation_id=0,action=dict(skill='HOLD'),evidence='synthetic test',
            choice_summary='hold',alternatives_summary='none',expected_result='finish')
        entry=dict(**request,simulation_start=0.,decision_wall_seconds=1.2,tokens=None)
        match.decisions.append(entry);match.apply(request['action'],match.compile(request['action']));match.advance(2)
        for name in ('decisions','requests','observations'):(self.path/name).mkdir()
        self.data={'trial_config.json':cfg,'timeline.json':match.states,'events.json':[],
            'validation.json':match.validation(),'decisions.json':[entry],
            'decisions/000.json':entry,'requests/000.json':request,'observations/000.json':obs,
            'decisions/000_outcome.json':dict(simulation_end=2.,state=match.states[-1],execution_wall_seconds=.01)}
        self.save()

    def save(self):
        for name,value in self.data.items():(self.path/name).write_text(json.dumps(value))

    def test_readonly_loader_and_hashes(self):
        before={p.name:p.read_bytes() for p in self.path.glob('*.json')}
        cfg,states,events,validation,manifest=load_replay(self.path)
        self.assertFalse(manifest['policy_rerun']);self.assertEqual(len(manifest['source_sha256']),4)
        self.assertEqual(states[-1]['t'],2.)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.path.glob('*.json')})

    def test_unfinished_rejected(self):
        self.data['timeline.json'][-1]['finished']=None;self.save()
        with self.assertRaises(ValueError):load_replay(self.path)

    def test_tick_gap_rejected(self):
        self.data['timeline.json'].pop(2);self.save()
        with self.assertRaises(ValueError):load_replay(self.path)

    def test_score_mismatch_rejected(self):
        self.data['validation.json']['score']=99;self.save()
        with self.assertRaises(ValueError):load_replay(self.path)

    def test_commit_outcome_audit(self):
        result,*_=audit(self.path)
        self.assertEqual(result['status'],'passed');self.assertIsNone(result['token_usage'])
        self.assertEqual(result['latency_seconds']['total'],1.2)

    def test_modified_request_rejected(self):
        self.data['requests/000.json']=copy.deepcopy(self.data['requests/000.json'])
        self.data['requests/000.json']['action']['skill']='SCAN';self.save()
        with self.assertRaises(AssertionError):audit(self.path)

    def test_modified_outcome_rejected(self):
        self.data['decisions/000_outcome.json']=copy.deepcopy(self.data['decisions/000_outcome.json'])
        self.data['decisions/000_outcome.json']['state']['score']=9;self.save()
        with self.assertRaises(AssertionError):audit(self.path)


if __name__=='__main__':unittest.main()
