"""Run the frozen D0 pilot locally without HTTP or a model credential."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

from llm_audit_observation_control import DirectClient, NoModel, PolicyError
from llm_decision_service import ServiceState, code_version
from llm_option_policy import LLMOptionPolicy
from llm_replay import replay
from llm_run_episode import option_log
from llm_study_data import emit, read_jsonl
from llm_study_failures import diagnose
from llm_study_online import HERE, config_matrix, load_plan, wilson
from llm_study_protocol import digest
from option_env import run_episode


class BoundaryIntervention(DirectClient):
    def __init__(self,state,expected,action):
        super().__init__(state)
        self.expected=expected
        self.action=action
        self.applied=None

    def decide(self,request):
        result=super().decide(request)
        reference=self.expected['request']
        if request['request_id']!=reference['request_id']:
            return result
        original=self.expected['response']
        if (self.applied is not None or request!=reference or result['status']!='boundary' or
                original['status']!='boundary' or result['action']!=original['action'] or
                not request['public']['action_mask'][self.action]):
            raise ValueError('Intervention failed to reproduce the recorded public boundary')
        self.applied=dict(request_id=request['request_id'],time=request['time'],
                          stage=request['stage'],recorded_action=result['action'],
                          substituted_action=self.action)
        return dict(result,action=self.action)


def execute(first_seed, second_seed, directory, config):
    episode_id = directory.name
    state = ServiceState(config, directory)
    no_model = NoModel()
    state.orchestrator.reasoner = no_model
    policy = LLMOptionPolicy(episode_id, DirectClient(state), 48.)
    policy.record_referee_telemetry = True
    started = time.perf_counter()
    row = dict(episode_id=episode_id, first_seed=first_seed, second_seed=second_seed,
               summary=None, error=None)
    try:
        summary, records = run_episode(first_seed, second_seed, policy)
        policy.finish(summary)
        row.update(status='completed', summary=summary, options=option_log(records))
    except PolicyError as error:
        if error.response['error_type'] != 'PolicyAbstention':
            raise
        row.update(status='policy_failure', error=error.response,
                   options=option_log(getattr(error, 'closed_options', [])),
                   referee_telemetry=getattr(error, 'referee_telemetry', None))
    row.update(wall_seconds=time.perf_counter()-started, external_model_calls=no_model.calls)
    if no_model.calls:
        raise AssertionError('The local D0 check entered the model interface')
    checked = replay(state.log_path, directory/'replay.json', config)
    row.update(requests_replayed=checked['same_decisions'], requests_logged=checked['count'],
               formal_terminal_replayed=bool(checked['finishes']) and all(
                   result['same_evaluation'] for result in checked['finishes']))
    if (row['requests_replayed'] != row['requests_logged'] or
            (row['status']=='completed') != row['formal_terminal_replayed']):
        raise AssertionError(f'D0 replay mismatch in {episode_id}')
    emit(directory/'episode.json', dict(row, own_trajectory=policy.own_trajectory))
    return row


def run(plan_path, output):
    plan = load_plan(plan_path, 'pilot')
    config = config_matrix('M1')['D0']
    output.mkdir(parents=True, exist_ok=False)
    report_path = output/'numeric_local.json'
    report = dict(schema='llm-study-local-numeric-v1', seed_plan=plan,
                  seed_plan_sha256=digest(plan), config=config,
                  code_version=code_version(), platform=platform.platform(),
                  python=platform.python_version(), rows=[],
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Local D0-only engineering pilot. Run D0 again in the Ubuntu '
                        'paired batch. Keep local results out of model selection and '
                        'cross-machine latency and head-to-head comparisons.')
    emit(report_path, report)
    for index, pair in enumerate(plan['seed_pairs']):
        row = execute(*pair, output/f'pair-{index+1:03d}', config)
        report['rows'].append(row)
        emit(report_path, report)
        print(json.dumps(dict(index=index+1, status=row['status'],
                              terminal=(row['summary'] or {}).get('reason'))), flush=True)
    n = len(report['rows'])
    wins = sum((row['summary'] or {}).get('winner') == 'attacker' for row in report['rows'])
    terminals = Counter((row['summary'] or {}).get('reason', row['status'])
                        for row in report['rows'])
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                  attempted=n, wins=wins, terminal_counts=dict(terminals),
                  wilson_95=wilson(wins,n),
                  external_model_calls=sum(row['external_model_calls'] for row in report['rows']))
    emit(report_path, report)
    print(json.dumps({k: report[k] for k in ('attempted','wins','terminal_counts','wilson_95',
                                             'external_model_calls')}), flush=True)


def audit_failures(output):
    report = json.loads((output/'numeric_local.json').read_text())
    if not report.get('complete') or report['schema']!='llm-study-local-numeric-v1':
        raise ValueError('Expected a completed local D0 run')
    cases = []
    for row in report['rows']:
        if (row['summary'] or {}).get('winner') == 'attacker':
            continue
        directory = output/row['episode_id']
        recorded = json.loads((directory/'episode.json').read_text())
        episode = dict(recorded, summary=recorded['summary'] or {})
        events = read_jsonl((directory/'decision_events.jsonl').read_text())
        result = diagnose(episode,dict(row,failure_category=(row['error'] or {}).get(
                          'error_type')),events)
        cases.append(dict(episode_id=row['episode_id'], result=result))
        print(json.dumps(dict(episode_id=row['episode_id'],
              recorded_tag_reproduced=result.get('reproduced_recorded_tag'),
              survived_selected_option=result.get('survived_selected_option'))),flush=True)
    target = output/'numeric_failure_diagnostics.json'
    emit(target,dict(scope='Local D0 one-Option outcomes; later play remains untested',
                     seed_plan_sha256=report['seed_plan_sha256'], cases=cases))
    return cases


def follow_through(output):
    report=json.loads((output/'numeric_local.json').read_text())
    if not report.get('complete'):
        raise ValueError('Expected a completed local D0 run')
    root=output/'single_wait_follow_through'
    root.mkdir(exist_ok=False)
    rows=[]
    for original in report['rows']:
        summary=original['summary'] or {}
        if summary.get('reason')!='tag' or original['options'][-1]['action']!='NEXT_COVER_OR_GOAL':
            continue
        directory=output/original['episode_id']
        start=original['options'][-1]['start']
        events=read_jsonl((directory/'decision_events.jsonl').read_text())
        expected=next(event for event in events if event['kind']=='decision' and
                      abs(event['request']['time']-start)<1e-5)
        state=ServiceState(report['config'],root/original['episode_id'])
        no_model=NoModel()
        state.orchestrator.reasoner=no_model
        client=BoundaryIntervention(state,expected,action=4)
        policy=LLMOptionPolicy(original['episode_id'],client,48.)
        policy.record_referee_telemetry=True
        try:
            alternative,options=run_episode(original['first_seed'],original['second_seed'],policy)
            policy.finish(alternative)
            row=dict(episode_id=original['episode_id'],status='completed',summary=alternative,
                     options=option_log(options))
        except PolicyError as error:
            if error.response['error_type']!='PolicyAbstention':
                raise
            row=dict(episode_id=original['episode_id'],status='policy_failure',
                     summary=None,error=error.response,
                     options=option_log(getattr(error,'closed_options',[])))
        if client.applied is None or no_model.calls:
            raise AssertionError('Expected one local Skill intervention and zero model calls')
        row.update(original_summary=summary,intervention=client.applied,
                   external_model_calls=no_model.calls)
        rows.append(row)
        emit(state.log_dir/'episode.json',row)
        print(json.dumps(dict(episode_id=row['episode_id'],status=row['status'],
                              summary=row['summary'])),flush=True)
    emit(root/'summary.json',dict(rows=rows,analysis='Retrospective one-shot WAIT then D0 on '
         'all local D0 tagging losses occurring in the post-gate continuation Skill.'))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=HERE/'config'/'llm_study_pilot_seeds.json')
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--audit-existing-failures',action='store_true')
    mode.add_argument('--follow-through-after-wait',action='store_true')
    args = parser.parse_args()
    if args.audit_existing_failures:
        audit_failures(args.output)
    elif args.follow_through_after_wait:
        follow_through(args.output)
    else:
        run(args.plan,args.output)
