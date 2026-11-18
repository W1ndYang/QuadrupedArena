"""Compare patrol forecasts on identical public anchors and later legal sightings.

This is an offline diagnostic on the recorded LLM policy's trajectories. Future
sightings are used only for evaluation. They are action-conditioned, sparse,
overlapping samples, not independent ground-truth patrol-target labels.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import tarfile

from llm_compare_ubuntu import recorded_cache_from_comparison
from llm_decision_core import EpisodeContext,_compare_prediction
from llm_semantic import build_map,public_rules
from llm_trajectory import local_motion_envelope


def audit(source,output):
    provenance=recorded_cache_from_comparison(source)
    semantic_map=build_map();rules=public_rules()
    rows=[];forecasts=[]
    with tarfile.open(source) as outer:
        comparison=json.load(outer.extractfile('llm_comparison/comparison.json'))
        for index in range(comparison['count']):
            name=f'{index:03d}-llm'
            with tarfile.open(fileobj=io.BytesIO(outer.extractfile(
                    f'llm_comparison/{name}.tar.gz').read())) as inner:
                events=[json.loads(line) for line in inner.extractfile(
                    'llm_run/decision_events.jsonl').read().decode().splitlines()]
                config=json.load(inner.extractfile('llm_run/config_effective.json'))
            future={}
            for event in events:
                payload=event['request'] if event['kind']=='decision' else event.get('payload',{})
                for sample in payload.get('observations',[]):
                    future[(sample['stage'],sample['time'])]=sample
            context=EpisodeContext(name,semantic_map['map_id'],rules['version'],
                                   velocity_mode=config.get('velocity_mode','legacy'))
            for event in events:
                if event['kind']!='decision':continue
                request,response=event['request'],event['response']
                context.add_samples(request['observations'])
                snapshot=response.get('forecast',{})
                if snapshot.get('forecast_source')!='llm_paths':continue
                track=context.track_for(request['stage'])
                if track is None or track['id']!=snapshot['track_id']:
                    raise ValueError('Recorded forecast does not match reconstructed public track')
                traces,unknown=local_motion_envelope(semantic_map,track,request['time'],
                                                     snapshot['horizon'],rules)
                numeric=dict(snapshot,trajectories=traces,hypotheses=[],
                             forecast_source='paired_local_motion_envelope')
                count=missing=0
                for (stage,time),sample in sorted(future.items()):
                    lead=time-request['time']
                    if stage!=request['stage'] or not 0.<lead<=snapshot['horizon']:continue
                    a=_compare_prediction(snapshot,sample)
                    b=_compare_prediction(numeric,sample)
                    if a is None or b is None:
                        missing+=1;continue
                    count+=1
                    rows.append(dict(index=index,forecast_id=snapshot['forecast_id'],
                        generated_at=request['time'],observation_at=time,lead_seconds=lead,
                        llm_position_error_m=a['position_error_m'],
                        numeric_position_error_m=b['position_error_m'],
                        llm_coverage80=a['coverage80'],numeric_coverage80=b['coverage80']))
                forecasts.append(dict(index=index,forecast_id=snapshot['forecast_id'],
                    generated_at=request['time'],stage=request['stage'],paired_sightings=count,
                    unavailable_sightings=missing,numeric_unknown_mass=unknown))
    aggregate=[]
    for label,lo,hi in (('0-1s',0.,1.),('1-3s',1.,3.),('over-3s',3.,100.)):
        selected=[r for r in rows if lo<r['lead_seconds']<=hi]
        by_forecast={}
        for row in selected:by_forecast.setdefault(row['forecast_id'],[]).append(row)
        entry=dict(lead_bin=label,paired_sightings=len(selected),forecasts=len(by_forecast))
        for method in ('llm','numeric'):
            entry[method+'_sighting_mean_error_m']=(statistics.mean(
                r[method+'_position_error_m'] for r in selected) if selected else None)
            entry[method+'_forecast_balanced_mean_error_m']=(statistics.mean(
                statistics.mean(r[method+'_position_error_m'] for r in group)
                for group in by_forecast.values()) if by_forecast else None)
        aggregate.append(entry)
    report=dict(schema='paired-public-forecast-audit-v1',
        source_archive_sha256=provenance['source_archive_sha256'],
        source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
                       for n in ('llm_audit_forecasts.py','llm_decision_core.py','llm_trajectory.py')},
        note=__doc__,aggregate=aggregate,forecasts=forecasts,rows=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(aggregate=aggregate,forecasts=len(forecasts),
                          forecasts_with_sightings=sum(f['paired_sightings']>0 for f in forecasts)),indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.source,args.output)
