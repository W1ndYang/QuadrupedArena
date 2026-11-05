"""Three isolated DS LOW games. Stored accepted responses are never resampled."""
import argparse,statistics,json
import qwen_fixed_goal as storage
from ds_fixed_goal import R,read,dump,utc,scenario,save_match
from fixed_goal_probe import sha
from peek_braking_fix import SequenceProbe
from sequence_skills import validate_plan_output
import deepseek_client as ds
from qwen_client import usage_total

original_payload=ds.payload
def prompt_only_payload(config,messages):
    value=original_payload(config,messages);value.pop('response_format',None);return value
ds.payload=prompt_only_payload
storage.profile=lambda mode:read(R/'api_profile.json')
storage.validate_output=validate_plan_output

def run_one(number):
    for name,digest in read(R/'preregistration.json')['source_sha256'].items():assert sha(R/name)==digest,name
    case=read(R/'cases.json')[number-1];folder=R/f'run_{number}'
    if (folder/'result.json').exists():print('ALREADY_COMPLETE',number,flush=True);return
    scenario(case['layout']);cfg=read(folder/'trial_config.json');base,rules=cfg['base'],cfg['rules']
    storage.prompt=lambda b,r:(R/'prompts'/f"{case['layout']}.txt").read_text().removesuffix('\n')
    planner=storage.QwenPlanner(folder,base,rules,'low',client=ds.call_api,key=ds.load_key())
    (folder/'system_prompt.txt').write_text(planner.system['content']+'\n')
    probe=SequenceProbe(base,rules,planner,folder);probe.metadata.update(attacker_policy='deepseek_flash_low',architecture='peek-sequences-v0.4.1-braking')
    dump(folder/'process.json',dict(status='running',utc=utc()))
    try:
        result=probe.run(f"deepseek_low_{case['layout']}_v041");save_match(folder,result)
        dump(folder/'timeline.json',result['states']);dump(folder/'events.json',result['events'])
        summary=result['summary'];summary.update(status='complete',thinking_mode='low',run=number,layout=case['layout'],
            usage=usage_total(planner.records),api_total_seconds=sum(r['api_wall_seconds'] for r in planner.records),
            api_median_seconds=statistics.median(r['api_wall_seconds'] for r in planner.records),
            response_models=sorted({r['response_model'] for r in planner.records}),completion_mode='unassisted')
        failures=list(folder.glob('api_calls/*_failure.json'))
        if failures:summary.update(completion_mode='network_resumed',failed_attempt_usage_unknown=True,incomplete_attempts=len(failures))
        dump(folder/'result.json',summary);dump(folder/'process.json',dict(status='complete',utc=utc()))
        print('MATCH_COMPLETE',number,json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as e:
        dump(folder/'partial_timeline.json',probe.trace);dump(folder/'skill_events.json',probe.controllers[0].events)
        dump(folder/'process.json',dict(status='interrupted',error_type=type(e).__name__,simulation_seconds=probe.trace[-1]['t'] if probe.trace else 0,utc=utc()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=int,choices=[1,2,3]);p.add_argument('--execute-paid',action='store_true');a=p.parse_args()
    if not a.execute_paid:raise ValueError('--execute-paid required')
    for i in ([a.run] if a.run else [1,2,3]):run_one(i)
