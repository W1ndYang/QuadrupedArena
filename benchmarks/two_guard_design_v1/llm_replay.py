"""Re-run saved public requests without starting the simulation."""
import argparse
import json
import math
from pathlib import Path

from llm_motion_client import APIReasoner
from llm_study_architecture import make_orchestrator


def evaluation_matches(left,right):
    """Keep structure/counts exact; tolerate floating summation roundoff only."""
    if type(left)!=type(right):return False
    if isinstance(left,dict):
        return (left.keys()==right.keys() and
                all(evaluation_matches(left[key],right[key]) for key in left))
    if isinstance(left,list):
        return len(left)==len(right) and all(evaluation_matches(a,b) for a,b in zip(left,right))
    if isinstance(left,float):return math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12)
    return left==right


class RecordedReasoner:
    def __init__(self,events):
        self.responses={}
        for event in events:
            if event['kind']!='decision':
                continue
            response=event['response']
            recorded=(event.get('llm_response') or
                      response.get('reasoning',{}).get('raw_response'))
            if recorded is not None:
                self.responses[event['request']['request_id']]=recorded

    def infer(self,messages):
        request=json.loads(messages[-1]['content'])['decision']
        request_id=request['request_id']
        if request_id not in self.responses:
            raise ValueError(f'No recorded LLM response for {request_id}')
        return self.responses[request_id]


def _effective_config(events, supplied, live_model):
    if live_model:
        return supplied, 'supplied_live_model'
    recorded=[event['config'] for event in events
              if event.get('kind')=='decision' and 'config' in event]
    if not recorded:
        return supplied, 'supplied_legacy_log'
    if any(row!=recorded[0] for row in recorded[1:]):
        raise ValueError('Recorded decisions used inconsistent service configurations')
    return recorded[0], 'recorded_service'


def replay(input_log,output,config,live_model=False):
    events=[json.loads(line) for line in Path(input_log).read_text().splitlines()
            if line.strip()]
    effective_config,config_source=_effective_config(events,config,live_model)
    reasoner=(APIReasoner(effective_config) if live_model else RecordedReasoner(events))
    orchestrator=make_orchestrator(reasoner,effective_config)
    rows=[];finishes=[]
    for event in events:
        if event['kind']=='finish':
            payload=event['payload']
            finished=orchestrator.finish(payload['episode_id'],payload['summary'],
                                          payload.get('observations',[]))
            finishes.append(dict(episode_id=payload['episode_id'],
                replayed_evaluation=finished['evaluation'],
                original_evaluation=event['response']['evaluation'],
                exact_evaluation=(finished['evaluation']==event['response']['evaluation']),
                same_evaluation=evaluation_matches(finished['evaluation'],
                                                  event['response']['evaluation']),
                float_tolerance=dict(relative=1e-12,absolute=1e-12)))
            continue
        if event['kind']!='decision':continue
        request=event['request']
        original=event['response']
        if (not live_model and event.get('llm_request') and
                request['request_id'] not in reasoner.responses):
            rows.append(dict(episode_id=request['episode_id'],
                request_id=request['request_id'],time=request['time'],
                original_status=original['status'],replayed_status='unreplayable',
                same_decision=False,replayable=False,
                reason='Original LLM response was not recorded'))
            continue
        try:
            response=orchestrator.decide(request)
        except Exception as exc:
            response=dict(status='error',error_type=type(exc).__name__,
                          error=str(exc))
        same=(original['status']==response['status'])
        if original['status']=='error':
            same=(same and original.get('error_type')==response.get('error_type')
                  and original.get('error')==response.get('error'))
        else:
            same=(same and original.get('action')==response.get('action')
                  and original.get('plan')==response.get('plan'))
        rows.append(dict(episode_id=request['episode_id'],
            request_id=request['request_id'],time=request['time'],
            original_status=original['status'],replayed_status=response['status'],
            original_action=original.get('action'),replayed_action=response.get('action'),
            original_plan=original.get('plan'),replayed_plan=response.get('plan'),
            original_error=original.get('error'),replayed_error=response.get('error'),
            same_decision=same,replayable=True))
    report=dict(mode='live_model' if live_model else 'recorded_llm',
                config_source=config_source,
                count=len(rows),same_decisions=sum(x['same_decision'] for x in rows),
                replayable_count=sum(x['replayable'] for x in rows),
                unreplayable_request_ids=[x['request_id'] for x in rows
                                          if not x['replayable']],
                public_requests_only=True,rows=rows,finishes=finishes)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--live-model',action='store_true',
                        help='Calls paid API using configured key; default reuses saved responses')
    args=parser.parse_args()
    result=replay(args.input_log,args.output,
                  json.loads(args.config.read_text()),args.live_model)
    print(json.dumps(dict(mode=result['mode'],count=result['count'],
                          same_decisions=result['same_decisions'],
                          finish_count=len(result['finishes']),
                          matching_finishes=sum(x['same_evaluation'] for x in
                                                result['finishes']),
                          config_source=result['config_source'])))


if __name__=='__main__':main()
