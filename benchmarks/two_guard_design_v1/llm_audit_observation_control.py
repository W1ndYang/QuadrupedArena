"""No-API causal diagnostic: change observation budgets, keep numerical planning.

Returned seeds are post-hoc mechanism checks, not new held-out evidence. Fresh
seeds and all three budgets are fixed before any simulation; no policy tuning
or best-case selection is performed. No saved guard measurements are replayed
into a counterfactual rollout: every run obtains its own legal observations.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

from llm_compare_ubuntu import recorded_cache_from_comparison
from llm_decision_service import ServiceState
from llm_option_policy import LLMOptionPolicy
from online_env import ACTION_NAMES
from option_env import run_episode


class NoModel:
    def __init__(self):self.calls=0
    def infer(self,messages):
        self.calls+=1
        raise AssertionError('This numerical control must never call a model')


class PolicyError(RuntimeError):
    def __init__(self,response):
        self.response=response
        super().__init__(response.get('error','Decision failed'))


class DirectClient:
    def __init__(self,state):self.state=state
    def decide(self,request):
        response=self.state.decide(request)
        if response['status']=='error':raise PolicyError(response)
        return response
    def finish(self,payload):return self.state.finish(payload)


def options(records):
    return [dict(stage=row['stage'],action=ACTION_NAMES[row['action']],
        start=row['time'],end=row['end_time'],duration_seconds=row['elapsed_seconds'],
        termination=row['termination_reason']) for row in records]


def audit(source,output,fresh_count=16):
    # The importer validates the checksums of both archive levels. Its model
    # responses are deliberately not supplied to the numerical control.
    provenance=recorded_cache_from_comparison(source)
    source_episodes={}
    with tarfile.open(source) as outer:
        comparison=json.load(outer.extractfile('llm_comparison/comparison.json'))
        for i in range(comparison['count']):
            name=f'{i:03d}-llm'
            with tarfile.open(fileobj=io.BytesIO(outer.extractfile(
                    f'llm_comparison/{name}.tar.gz').read())) as inner:
                source_episodes[i]=json.load(inner.extractfile(f'llm_run/{name}.json'))
        with tarfile.open(fileobj=io.BytesIO(outer.extractfile(
                'llm_comparison/000-numeric.tar.gz').read())) as inner:
            config=json.load(inner.extractfile('llm_run/config_effective.json'))
    if comparison['fixture_only']:
        raise ValueError('Use a real returned comparison, not a fixture study')
    config={k:v for k,v in config.items() if k not in
            ('api_key','api_key_file','recorded_response_cache_file')}
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    pairs=dict(returned=comparison['seed_pairs'],
               fresh=[[1010100+17*i,1020100+19*i] for i in range(fresh_count)])
    budgets=[[1,1],[1,2],[2,2]]
    report=dict(schema='llm-observation-control-audit-v1',
        started_utc=datetime.now(timezone.utc).isoformat(),source_archive=str(source),
        source_archive_sha256=provenance['source_archive_sha256'],
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ('llm_audit_observation_control.py','llm_decision_core.py',
                                    'llm_option_policy.py','option_env.py')},
        groups=pairs,observation_budgets=budgets,duration=comparison['duration'],
        external_model_calls=0,fixture_only=False,rows=[],
        note='Post-hoc returned-case mechanism diagnostic; fresh numerical-only probe. '
             'No fresh LLM comparison. Stage minima use the existing legal observation '
             'selector; other information gates, per-window caps and optimizer unchanged.')
    target=output/'summary.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    for group,seeds in pairs.items():
        for budget in budgets:
            label='-'.join(map(str,budget))
            for index,(first,second) in enumerate(seeds):
                episode_id=f'{group}-{label}-{index:03d}'
                directory=output/episode_id
                settings=dict(config,max_llm_calls_per_stage=0,minimum_peeks_by_stage=budget)
                state=ServiceState(settings,directory,fixture=False)
                forbidden=NoModel();state.orchestrator.reasoner=forbidden
                policy=LLMOptionPolicy(episode_id,DirectClient(state),comparison['duration'])
                row=dict(group=group,budget=budget,index=index,first_seed=first,second_seed=second)
                start=time.perf_counter()
                try:
                    summary,records=run_episode(first,second,policy,comparison['duration'])
                    policy.finish(summary)
                    row.update(status='completed',summary=summary,options=options(records))
                except PolicyError as exc:
                    if exc.response.get('error_type')!='PolicyAbstention':raise
                    row.update(status='policy_failure',summary=dict(winner='defender',
                        reason='policy_abstention',seconds=policy.last_time,
                        synthetic_result=True,formal_terminal_reached=False),error=exc.response)
                if forbidden.calls:raise AssertionError('Unexpected model invocation')
                row.update(external_model_calls=forbidden.calls,wall_seconds=time.perf_counter()-start,
                           own_trajectory=policy.own_trajectory)
                if group=='returned' and row['status']=='completed':
                    reference=source_episodes[index]
                    row['same_options_as_llm']=reference.get('options')==row['options']
                    trace=reference.get('own_trajectory',[])
                    current=policy.own_trajectory
                    aligned=(len(trace)==len(current) and all(
                        a['time']==b['time'] and a['stage']==b['stage'] for a,b in zip(trace,current)))
                    error=(max(abs(a['xy'][axis]-b['xy'][axis])
                               for a,b in zip(trace,current) for axis in (0,1)) if aligned else None)
                    row['matched_trace_max_position_error_m']=error
                report['rows'].append(row)
                target.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in
                    ('own_trajectory','options','error')},ensure_ascii=False),flush=True)
    report['aggregate']={}
    for group,seeds in pairs.items():
        report['aggregate'][group]={}
        for budget in budgets:
            rows=[r for r in report['rows'] if r['group']==group and r['budget']==budget]
            report['aggregate'][group]['-'.join(map(str,budget))]=dict(
                attempted=len(rows),wins=sum(r['summary']['winner']=='attacker' for r in rows),
                policy_failures=sum(r['status']=='policy_failure' for r in rows))
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['aggregate']),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fresh-count',type=int,default=16)
    args=parser.parse_args()
    if not 0<=args.fresh_count<=1000:parser.error('--fresh-count must be 0..1000')
    audit(args.source,args.output_dir,args.fresh_count)
