"""Qwen mode comparison using the exact frozen fixed-goal skills and referee."""
import argparse
import copy
import json
import statistics
from pathlib import Path

from ds_fixed_goal import R,scenario,prompt,validate_output,digest,read,dump,utc,SkillProbe,save_match,FIELDS
from qwen_client import call_api,load_key,usage_total

CASES=read(R/'cases.json')

def profile(mode):return read(R/'profiles'/f'thinking_{mode}.json')

def parse_final(content,index):
    text=content.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):text=text[8:-4]
    value=json.loads(text);normalization=None
    # Historical Qwen adapter compatibility: no control or assessment text changes.
    if isinstance(value,dict) and isinstance(value.get('assessment'),list):
        values=value['assessment']
        if len(values)==4 and all(isinstance(v,str) for v in values):
            value['assessment']=dict(zip(FIELDS,values))
            normalization=dict(kind='ordered_assessment_array_to_named_object',fields=list(FIELDS),control_fields_changed=False,assessment_strings_changed=False)
    return validate_output(json.dumps(value,ensure_ascii=False,allow_nan=False),index),normalization

class QwenPlanner:
    def __init__(self,folder,base,rules,mode,client=call_api,key=None):
        self.folder=folder;self.profile=profile(mode);self.client=client
        self.key=load_key() if key is None else key
        self.system=dict(role='system',content=prompt(base,rules));self.records=[];self.actions=[]
        (folder/'api_calls').mkdir(parents=True,exist_ok=True);(folder/'accepted').mkdir(exist_ok=True)
        dump(folder/'api_profile.json',self.profile)

    def __call__(self,packet):
        index=packet['observation_id'];stem=self.folder/'api_calls'/f'{index:03d}'
        history=[{k:a[k] for k in ('observation_id','skill','params')} for a in self.actions[-4:]]
        request=dict(system=self.system,observation=packet,recent_decisions=history,profile=self.profile)
        request_hash=digest(request);rp=Path(str(stem)+'_request.json');sp=Path(str(stem)+'_response.json');ap=self.folder/'accepted'/f'{index:03d}.json'
        if rp.exists():
            if digest(read(rp))!=request_hash:raise ValueError('Saved request differs; refusing decision resampling')
        else:dump(rp,request)
        if sp.exists():
            response=read(sp);print('REUSE_RESPONSE',self.folder.name,index,flush=True)
        else:
            print('API_START',self.folder.name,index,packet['simulation_seconds'],utc(),flush=True)
            try:
                response=self.client(self.profile,self.key,[self.system,dict(role='user',content=json.dumps(dict(observation=packet,recent_decisions=history),ensure_ascii=False))])
            except Exception as e:
                dump(Path(str(stem)+'_failure.json'),dict(utc=utc(),error_type=type(e).__name__,request_sha256=request_hash,complete_response_saved=False))
                raise
            dump(sp,response)
            print('API_END',self.folder.name,index,round(response['api_wall_seconds'],3),response.get('usage'),flush=True)
        self.records.append(response);dump(self.folder/'api_usage.json',usage_total(self.records))
        if response['finish_reason']!='stop' or response['tool_calls']:raise ValueError('Response incomplete or used tools')
        action,normalization=parse_final(response['content'],index)
        if normalization:dump(Path(str(stem)+'_normalization.json'),normalization)
        if ap.exists():assert read(ap)['action']==action
        else:dump(ap,dict(action=action,request_sha256=request_hash,simulation_seconds=packet['simulation_seconds'],normalization=normalization))
        self.actions.append(action);return action

class QwenProbe(SkillProbe):
    def __init__(self,base,rules,decider,folder,mode,pair):
        super().__init__(base,rules,decider,folder)
        self.mode=mode;self.pair=pair;self.metadata['attacker_policy']='qwen_thinking_'+mode
    def run(self,identifier):
        result=super().run(identifier)
        result['summary'].update(condition='qwen_goal_only',thinking_mode=self.mode,pair=self.pair)
        return result

def run_one(number,paid=False):
    if not paid:raise ValueError('Explicit --execute-paid required')
    case=CASES[number-1];base,rules=scenario(case['layout']);folder=R/f'run_{number}';folder.mkdir(exist_ok=True)
    from ds_fixed_goal import WAYPOINTS
    dump(folder/'trial_config.json',dict(base=base,rules=rules,waypoints=WAYPOINTS))
    dump(folder/'case.json',case)
    (folder/'system_prompt.txt').write_text(prompt(base,rules)+'\n')
    decider=QwenPlanner(folder,base,rules,case['mode']);probe=QwenProbe(base,rules,decider,folder,case['mode'],case['pair'])
    try:
        result=probe.run(f'qwen_{case["mode"]}_pair_{case["pair"]}')
        save_match(folder,result);dump(folder/'timeline.json',result['states']);dump(folder/'events.json',result['events'])
        summary=result['summary'];summary.update(usage=usage_total(decider.records),
            api_median_seconds=statistics.median(r['api_wall_seconds'] for r in decider.records),
            api_total_seconds=sum(r['api_wall_seconds'] for r in decider.records),
            response_models=sorted(set(r['response_model'] for r in decider.records)),status='complete',run=number)
        dump(folder/'result.json',summary);dump(folder/'process.json',dict(status='complete',utc=utc()))
        print('MATCH_COMPLETE',number,json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as e:
        dump(folder/'partial_timeline.json',probe.trace)
        dump(folder/'process.json',dict(status='interrupted',error_type=type(e).__name__,simulation_seconds=probe.trace[-1]['t'] if probe.trace else 0,utc=utc()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=int,choices=range(1,7),required=True);p.add_argument('--execute-paid',action='store_true');args=p.parse_args();run_one(args.run,args.execute_paid)
