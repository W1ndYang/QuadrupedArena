"""Small, paired CPU-only policy comparison. Not a training or balance benchmark."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
from adaptive_motion import load_config,build


def evaluate(seeds):
    config=load_config();rows=[]
    for seed in seeds:
        for mode in ('nearest_point','risk_aware'):
            cfg=copy.deepcopy(config);cfg['seed']=seed
            states,events,validation=build(cfg,quadruped_mode=mode);last=states[-1]
            rows.append(dict(seed=seed,mode=mode,score=last['score'],duration=last['t'],hp_a=last['hp'][0],
                p2_score=last['target_scores'][1],finish=last['finished'],
                hits_received=sum(e['type']=='hit' and e['attacker']==1 for e in events),
                exposure_seconds=validation['a_exposed_seconds'],
                exposure_fraction=validation['a_exposed_seconds']/max(last['t'],cfg['dt']),
                min_clearance=validation['min_center_to_cover'],min_separation=validation['min_actor_separation']))
            print(json.dumps(rows[-1]),flush=True)
    summary={}
    for mode in ('nearest_point','risk_aware'):
        values=[r for r in rows if r['mode']==mode]
        summary[mode]={key:statistics.mean(r[key] for r in values) for key in ('score','duration','hp_a','p2_score','hits_received','exposure_seconds','exposure_fraction')}
        summary[mode]['survived_count']=sum(r['hp_a']>0 for r in values)
    source=Path(__file__).with_name('adaptive_motion.py')
    return dict(seeds=seeds,config=config,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,summary=summary,
        limitations=['Small exploratory sample, not statistical proof','Both groups use identical visibility-limited humanoid',
        'A differs in waypoint selection, risk planning, evasion, facing and reverse execution; not an isolated risk-weight ablation',
        'Same seed gives paired random streams, but different aim episodes consume draws differently'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20261021,20261022,20261023,20261024,20261025])
    args=parser.parse_args();result=evaluate(args.seeds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))
