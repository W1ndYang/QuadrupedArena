"""Offline first-stage ablation on saved LLM output and the formal executor.

No API calls. A missing saved response stops that case explicitly. Reused
responses were generated under their original prompts/horizons; this audit
tests numerical/execution changes, not how a live LLM would respond anew.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import time

from llm_decision_service import ServiceState
from llm_option_policy import LLMOptionPolicy
from llm_semantic import COMPACT_SCHEMA_VERSION
from option_env import run_episode


class StageComplete(Exception):pass
class MissingResponse(Exception):pass


def load_archive(path):
    with tarfile.open(path) as archive:
        manifest=json.load(archive.extractfile('llm_run/run_manifest.json'))
        config=json.load(archive.extractfile('llm_run/config_effective.json'))
        events=[json.loads(line) for line in
                archive.extractfile('llm_run/decision_events.jsonl').read().decode().splitlines()]
    return manifest,config,events


class RecordedReasoner:
    def __init__(self,events):
        self.events=events;self.calls=0
    def infer(self,messages):
        decision=json.loads(messages[-1]['content'])['decision']
        matches=[event for event in self.events if event['kind']=='decision' and
                 event['request']['stage']==(0 if decision['region']=='first' else 1) and
                 abs(event['request']['time']-decision['game_time'])<.051]
        if not matches:raise MissingResponse(f"No saved output at game time {decision['game_time']}")
        event=matches[0];response=event['response']
        raw=event.get('llm_response') or response.get('reasoning',{}).get('raw_response')
        if raw is None:raise MissingResponse('Matching event has no model output')
        self.calls+=1
        result=copy.deepcopy(raw);value=json.loads(result['text'])
        # The previous wire-format failure can be studied using its valid edge
        # path, explicitly discarding the redundant MOVE labels.
        if value['schema']!=COMPACT_SCHEMA_VERSION:
            value['schema']=COMPACT_SCHEMA_VERSION
            for hypothesis in value['hypotheses']:
                hypothesis['dwell_node_ids']=list(dict.fromkeys(
                    event['at_id'] for event in hypothesis.pop('events')
                    if event['type']=='DWELL'))
                hypothesis.pop('next_target_id');hypothesis.pop('uncertainty')
        value['track_id']=decision['track_id']
        result['text']=json.dumps(value)
        return result


class DirectClient:
    def __init__(self,state,force_first_north=False):
        self.state=state;self.force_first_north=force_first_north;self.interventions=[]
    def decide(self,request):
        response=self.state.decide(request)
        if response['status']=='error':
            if response.get('error_type')=='MissingResponse':raise MissingResponse(response['error'])
            raise RuntimeError(response['error'])
        if self.force_first_north and request['time']==3.3 and not self.interventions:
            plan=dict(side='north',wait=0.,smoke=False)
            self.interventions.append(dict(time=3.3,forced_plan=plan,
                                           computed_status=response['status']))
            return dict(episode_id=request['episode_id'],request_id=request['request_id'],
                        status='plan',plan=plan,boundary_recheck=True)
        return response


class AuditPolicy(LLMOptionPolicy):
    def __init__(self,*args):super().__init__(*args);self.self_trace=[]
    def observe(self,obs,mask):
        super().observe(obs,mask)
        self.self_trace.append(dict(time=self.last_time,stage=self.stage,
            xy=[float(obs[self.index['self_x']])*12.,float(obs[self.index['self_y']])*8.]))
        if self.stage==1:raise StageComplete()


def audit(archive_paths,output,selected_cases=None):
    manifest,config,events=load_archive(archive_paths[0])
    for path in archive_paths[1:]:events+=load_archive(path)[2]
    cases=[('baseline',{}),
           ('guard_alignment',dict(trajectory_mode='observed_anchor_v2')),
           ('full_robot_path',dict(trajectory_mode='observed_anchor_v2',robot_rollout_mode='stateful_v2')),
           ('boundary_feedback',dict(trajectory_mode='observed_anchor_v2',robot_rollout_mode='stateful_v2',boundary_recheck=True)),
           ('bounded_no_boundary',dict(trajectory_mode='observed_anchor_v2',robot_rollout_mode='stateful_v2',
                max_peeks_per_stage=2,max_llm_calls_per_stage=1)),
           ('bounded_feedback',dict(trajectory_mode='observed_anchor_v2',robot_rollout_mode='stateful_v2',
                max_peeks_per_stage=2,max_llm_calls_per_stage=1,boundary_recheck=True)),
           ('numerical_only',dict(trajectory_mode='observed_anchor_v2',robot_rollout_mode='stateful_v2',
                max_peeks_per_stage=2,max_llm_calls_per_stage=0,boundary_recheck=True)),
           ('numerical_two_peeks',dict(trajectory_mode='observed_anchor_v2',robot_rollout_mode='stateful_v2',
                minimum_peeks_per_stage=2,max_peeks_per_stage=2,max_llm_calls_per_stage=0,boundary_recheck=True)),
           ('boundary_rescue',dict(trajectory_mode='observed_anchor_v2',robot_rollout_mode='stateful_v2',
                max_peeks_per_stage=2,max_llm_calls_per_stage=1,boundary_recheck=True,audit_force_first_north=True))]
    cases=[(name,dict(changes,velocity_mode='recent_acceleration_v2'))
           if changes.get('trajectory_mode')=='observed_anchor_v2' else (name,changes)
           for name,changes in cases]
    if selected_cases:
        unknown=set(selected_cases)-{name for name,_ in cases}
        if unknown:raise ValueError(f'Unknown audit cases: {sorted(unknown)}')
        cases=[case for case in cases if case[0] in selected_cases]
    results=[]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,changes in cases:
        with tempfile.TemporaryDirectory() as directory:
            settings=dict(config,**changes)
            state=ServiceState(settings,directory,fixture=True)
            reasoner=RecordedReasoner(events);state.orchestrator.reasoner=reasoner
            client=DirectClient(state,changes.get('audit_force_first_north',False))
            policy=AuditPolicy(name,client,manifest['duration_seconds'])
            start=time.perf_counter();row=dict(case=name,changes=changes)
            try:
                summary,records=run_episode(manifest['first_seed'],manifest['second_seed'],policy,manifest['duration_seconds'])
                row.update(status='terminal',first_stage_pass=summary['handoff_time'] is not None,summary=summary)
            except StageComplete:
                row.update(status='first_stage_complete',first_stage_pass=True,seconds=policy.last_time)
            except MissingResponse as exc:
                row.update(status='needs_new_response',first_stage_pass=None,error=str(exc),seconds=policy.last_time)
            except RuntimeError as exc:
                row.update(status='runtime_error',first_stage_pass=None,error=str(exc),seconds=policy.last_time)
            row.update(wall_seconds=time.perf_counter()-start,recorded_outputs_used=reasoner.calls,
                       interventions=client.interventions,
                       decisions=policy.responses,self_trace=policy.self_trace)
            events_out=[json.loads(line) for line in state.log_path.read_text().splitlines()]
            row['decision_details']=[dict(time=e['request']['time'],status=e['response']['status'],
                timing=e['response'].get('timing'),optimizer=e['response'].get('optimizer'),
                gate=e['response'].get('gate')) for e in events_out]
            (output/f'{name}.json').write_text(json.dumps(row,indent=2)+'\n')
            results.append({key:value for key,value in row.items() if key not in ('self_trace','decision_details')})
            print(json.dumps(results[-1]),flush=True)
    report=dict(first_seed=manifest['first_seed'],second_seed=manifest['second_seed'],
        sources=[str(path) for path in archive_paths],paid_api_calls=0,
        source_archives=[dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                         for path in archive_paths],
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ('llm_decision_core.py','llm_trajectory.py','llm_robot_rollout.py',
                                    'llm_fast_optimizer.py','llm_option_policy.py','option_env.py',
                                    'llm_motion_client.py','llm_decision_service.py','llm_audit_alignment.py')},
        note='Recorded outputs under original prompts; first stage only, not held-out performance',cases=results)
    report['timing_note']='Cases share map/route caches; case wall times are diagnostics, not cold benchmarks'
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,action='append',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cases',nargs='+')
    args=parser.parse_args();audit(args.archive,args.output_dir,args.cases)
