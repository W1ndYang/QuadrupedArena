"""Gather supplemental legal observation histories with a fixed probe policy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from llm_decision_core import _check_public_request
from llm_decision_service import code_version
from llm_option_policy import LLMOptionPolicy
from llm_run_episode import option_log
from llm_study_data import emit
from llm_study_protocol import digest
from option_env import run_episode


HERE=Path(__file__).resolve().parent
PLAN=HERE/'config'/'llm_study_turn_collection.json'


class FixedProbeClient:
    def __init__(self,index):
        self.index=index
        self.requests=[]

    def decide(self,request):
        _check_public_request(request)
        public=request['public']
        if public['action_mask'][7]:
            action=4 if request['skill_state'].get('boundary_waits',0) < (self.index//2)%2 else 7
        elif public['peek_count']<2:
            action=2+(public['peek_count']+self.index//4+request['stage'])%2
        else:
            action=5+(self.index+request['stage'])%2
        if not public['action_mask'][action]:
            raise ValueError('Fixed probe attempted an illegal Option')
        response=dict(episode_id=request['episode_id'],request_id=request['request_id'],
                      action=action,status='skill')
        self.requests.append(dict(request=request,response=response))
        return response

    def finish(self,payload):
        return {'status':'recorded'}


class Collector(LLMOptionPolicy):
    def __init__(self,*args):
        super().__init__(*args)
        self.public_sightings=[]

    def observe(self,observation,mask):
        super().observe(observation,mask)
        if self.samples and self.samples[-1]['time']==self.last_time:
            current=self.samples[-1]
            if not self.public_sightings or current!=self.public_sightings[-1]:
                self.public_sightings.append(current)


def collect(output):
    plan=json.loads(PLAN.read_text())
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema=plan['schema'],split=plan['split'],seed_plan=plan,
                  seed_plan_sha256=digest(plan),code_version=code_version(),
                  started_utc=datetime.now(timezone.utc).isoformat(),episodes=[],
                  complete=False,external_model_calls=0)
    target=output/'collector_manifest.json'
    emit(target,manifest)
    for index,(first,second) in enumerate(plan['seed_pairs']):
        episode_id=f'event-dev-{index+1:03d}'
        client=FixedProbeClient(index)
        policy=Collector(episode_id,client,plan['duration'])
        summary,options=run_episode(first,second,policy,duration=plan['duration'])
        policy.finish(summary)
        log=output/f'{episode_id}-public.jsonl'
        with log.open('x') as handle:
            for observation in policy.public_sightings:
                handle.write(json.dumps(observation,allow_nan=False)+'\n')
        record=output/f'{episode_id}-decisions.json'
        emit(record,dict(episode_id=episode_id,requests=client.requests,
                         summary=summary,options=option_log(options)))
        manifest['episodes'].append(dict(episode_id=episode_id,seed_pair=[first,second],
            log=log.name, log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            decision_log=record.name, decision_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
            legal_sightings=len(policy.public_sightings)))
        emit(target,manifest)
        print(json.dumps(dict(completed=index+1,planned=len(plan['seed_pairs']),
                              sightings=len(policy.public_sightings))),flush=True)
    manifest.update(complete=True,finished_utc=datetime.now(timezone.utc).isoformat())
    emit(target,manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    collect(args.output)
