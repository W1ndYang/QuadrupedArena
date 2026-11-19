import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from llm_decision_service import ServiceState
from llm_option_policy import LLMOptionPolicy
from llm_replay import replay
from llm_study_architecture import screen
from llm_study_online import (METHODS, THINKING_LIMITS, batch_indices, config_matrix,
                              holm_primary, load_plan, maximum_calls, paired,
                              primary_comparisons, thinking_config_matrix)
from llm_bundle import create_bundle
from llm_compare_ubuntu import comparison_bundle, verify_comparison
from llm_study_data import emit
from llm_study_holdout import join
from llm_study_power import exact_power, freeze_holdout
from llm_study_protocol import digest, parse_direct, strict_json
from llm_study_turns import audit_case
from llm_semantic import ModelOutputError
from option_env import run_episode


HERE = Path(__file__).resolve().parent


class StudyTests(unittest.TestCase):
    def test_thinking_arms_change_only_the_mode_within_each_architecture(self):
        configs=thinking_config_matrix()
        base=config_matrix('M1')
        self.assertEqual(configs['D0'],base['D0'])
        self.assertEqual(len(configs),7)
        self.assertEqual(maximum_calls(base,16),832)
        self.assertEqual(maximum_calls(configs,16),1600)
        for method in ('D1_matched','D2','D3'):
            m1,m2=(configs[f'{method}_{model}'] for model in ('M1','M2'))
            self.assertEqual(m1,dict(base[method],**THINKING_LIMITS))
            self.assertEqual(m2,dict(m1,thinking={'type':'enabled'},reasoning_effort='high'))
        self.assertEqual(primary_comparisons(list(configs))[0],'D2_M2_vs_D2_M1')

    def test_holdout_batches_require_full_disjoint_seed_coverage(self):
        self.assertEqual([i for b in (1,2,3) for i in batch_indices(10,(b,3))],
                         list(range(10)))
        plan=dict(seed_pairs=[[800010,801010],[800020,801020]])
        methods=['D0','D2']
        reference=dict(phase='holdout',fixture_only=False,status='completed',
                       seed_plan=plan,seed_plan_sha256=digest(plan),methods=methods,
                       direct_model='M1',decision_runtime_version='fixture',
                       decision_config_sha256='fixture',git_head='fixture',
                       runner_sha256='fixture',duration_seconds=48.,
                       decision_clock='paused_simulation',
                       maximum_live_calls=0,started_utc='start',finished_utc='end')
        with TemporaryDirectory() as directory:
            root=Path(directory)
            paths=[]
            for index,(first,second) in enumerate(plan['seed_pairs']):
                batch=root/f'batch-{index}'
                batch.mkdir()
                rows=[]
                for method in methods:
                    row=dict(index=0,seed_index=index,method=method,
                             first_seed=first,second_seed=second,status='completed',
                             summary=dict(winner='attacker' if method=='D0' else 'defender',
                                          seconds=1.),
                             official_reason='goal' if method=='D0' else 'tag',
                             failure_category=None,formal_terminal=True,episode_wall_seconds=1.,
                             llm_inference_attempts=0,total_tokens_reported=0,safety_checks=[],
                             model_seconds=[],full_service_seconds=[],local_compute_seconds=[])
                    rows.append(row)
                    sub=batch/method
                    sub.mkdir()
                    emit(sub/'run_manifest.json',dict(fixture=True))
                    create_bundle(sub,batch/f'000-{method}.tar.gz')
                emit(batch/'comparison.json',dict(reference,holdout_batch=[index+1,2],
                    seed_indices=[index],seed_pairs=[[first,second]],row_count_planned=len(rows),
                    rows=rows))
                archive=root/f'batch-{index}.tar.gz'
                comparison_bundle(batch,archive)
                paths.append(archive)
            with self.assertRaises(ValueError):
                join(paths[:1],root/'incomplete')
            with self.assertRaises(ValueError):
                join(paths[:1]*2,root/'overlap')
            combined=join(paths[::-1],root/'combined')
            self.assertEqual(len(verify_comparison(combined)['files']),5)
            result=json.loads((root/'combined'/'comparison.json').read_text())
            self.assertEqual(result['aggregate']['paired']['D2_vs_D0']['reference_only'],2)

    def test_pairing_checks_and_api_sensitivity(self):
        rows = []
        for index, (a,b) in enumerate((('attacker','defender'),('defender','attacker'),
                                        ('attacker',None))):
            for method, winner in (('D0',a),('D2',b)):
                rows.append(dict(index=index,method=method,first_seed=10+index,
                                 second_seed=20+index,
                                 status='completed' if winner else 'api_failure',
                                 summary=dict(winner=winner) if winner else None))
        grouped = paired(rows,['D0','D2'],3)['D2_vs_D0']
        self.assertEqual((grouped['reference_only'],grouped['comparison_only']),(2,1))
        self.assertEqual(grouped['exact_two_sided_p'],1.)
        self.assertEqual(paired(rows,['D0','D2'],3,False)['D2_vs_D0']['n'],2)
        with self.assertRaises(ValueError):
            paired(rows+[rows[0]],['D0','D2'],3)
        with self.assertRaises(ValueError):
            paired([dict(rows[0],first_seed=100),*rows[1:]],['D0','D2'],3)
        self.assertEqual(holm_primary({'a':{'exact_two_sided_p':.02},
                                       'b':{'exact_two_sided_p':.03}}, ('a','b','c')),
                         {'a':.06,'b':.06,'c':None})

    def test_holdout_seed_generation_is_disjoint_and_frozen(self):
        pilot = load_plan(HERE/'config'/'llm_study_pilot_seeds.json','pilot')
        report = dict(suggested_common_seed_count=24,pilot_sha256='test-fixture',
                      pilot_seed_pairs=pilot['seed_pairs'],direct_model='M1',
                      methods=list(METHODS),decision_config_sha256='test-fixture',
                      decision_runtime_version='test-fixture')
        self.assertEqual(exact_power(4,.4,.15,.05),0.)
        with TemporaryDirectory() as temp:
            path = Path(temp)/'holdout.json'
            result = freeze_holdout(report,path)
            self.assertEqual(load_plan(path,'holdout'),result)
            other = freeze_holdout(report,Path(temp)/'copy.json')
            self.assertEqual(result['seed_pairs'],other['seed_pairs'])
            with self.assertRaises(FileExistsError):
                freeze_holdout(report,path)

    def test_visible_stop_then_turn_or_reverse(self):
        def trace(direction):
            xy = [[i*.05,0.] for i in range(4)] + [[.15,0.]]*5
            xy.extend([[.15+direction[0]*i*.05,direction[1]*i*.05]
                       for i in range(1,5)])
            return [dict(time=i/10,xy=pos,uncertainty_m=0.,frame_valid=True,stage=0)
                    for i,pos in enumerate(xy)]
        for direction, expected in (((0,1),'TURN'),((-1,0),'REVERSE'),((1,0),None)):
            with self.subTest(direction=direction):
                labels = [event['event'] for run in audit_case(trace(direction))
                          for event in run['events']]
                self.assertEqual(labels,[expected] if expected else [])
        self.assertFalse(any(run['events'] for run in audit_case(
            [point for i,point in enumerate(trace((-1,0))) if i!=7])))

    def test_direct_schema_and_hard_fallback(self):
        with self.assertRaises(ModelOutputError):
            strict_json('{"skill":"WAIT","skill":"SMOKE_SOUTH"}')
        req = {'stage': 0, 'public': {
            'action_mask': [False]*10, 'smoke_available': False, 'own_xy': [-8., 0.]}}
        decision = dict(allowed_skills=['WAIT'],
                        limits=dict(additional_peek_allowed=False, post_gate_waits_left=2,
                                    post_gate_retreat_available=True))
        result = parse_direct('{"schema":"skill-decision-v1","skill":"SMOKE_SOUTH",'
                              '"reason":"fixture","uncertainty":"fixture"}', ['SMOKE_SOUTH'])
        self.assertEqual(screen(result, req, decision, False)['status'], 'rejected')
        checked = screen(result, req, decision, True)
        self.assertEqual(checked['rejection'], 'skill_mask')
        self.assertEqual(checked['executed'], 'WAIT')

    def test_recorded_replay_for_every_architecture(self):
        configs = config_matrix('M1')
        seeds = load_plan(HERE/'config'/'llm_study_pilot_seeds.json', 'pilot')['seed_pairs'][0]
        messages = {}
        with TemporaryDirectory() as directory:
            for method in METHODS:
                with self.subTest(method=method):
                    path = Path(directory)/method
                    state = ServiceState(configs[method], path, fixture=True)
                    policy = LLMOptionPolicy('same-public-identity', state, 48.)
                    policy.record_referee_telemetry = True
                    summary, options = run_episode(*seeds, policy)
                    self.assertTrue(summary['referee_telemetry'])
                    policy.finish(summary)
                    result = replay(state.log_path, path/'replay.json', configs[method])
                    self.assertEqual(result['same_decisions'], result['count'])
                    self.assertTrue(result['finishes'][0]['same_evaluation'])
                    self.assertGreater(len(options), 1)
                    exchanges = [e for line in state.log_path.read_text().splitlines()
                                 if (e := json.loads(line)).get('llm_request')]
                    if method in ('D1_matched', 'D2', 'D3') and exchanges:
                        messages[method] = json.loads(exchanges[0]['llm_request'][-1]['content'])
                        self.assertNotIn('seed', messages[method]['decision'])
                        self.assertTrue(all(row['time'] <= exchanges[0]['request']['time']
                                            for row in messages[method]['decision']['observations']))
        self.assertEqual(messages['D2'], messages['D3'])
        self.assertEqual(messages['D1_matched'], messages['D2'])


if __name__ == '__main__':
    unittest.main()
