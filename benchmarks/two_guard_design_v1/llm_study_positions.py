"""Compare conditional position forecasts against the same legal later sightings."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import tarfile

from llm_bundle import verify_bundle
from llm_decision_core import _compare_prediction
from llm_study_data import emit, read_jsonl
from llm_study_offline import LOG, MODELS
from llm_study_protocol import digest
from llm_trajectory import generate, local_motion_envelope


def summarize(rows):
    summary=[]
    for model in MODELS:
        for lead in ('0-1s','1-3s','over-3s'):
            selected=[row for row in rows if row['model']==model and row['lead']==lead]
            groups=defaultdict(list)
            for row in selected:
                groups[row['sample_id']].append(row)
            result=dict(model=model,lead=lead,future_points=len(selected),forecasts=len(groups),
                        seed_groups=len({row['episode_id'] for row in selected}))
            for field in ('llm_position_error_m','numeric_position_error_m'):
                result[field]=(statistics.fmean(statistics.fmean(r[field] for r in group)
                    for group in groups.values()) if groups else None)
            summary.append(result)
    return summary


def analyze(source, data_path, output):
    verify_bundle(source)
    data=json.loads(data_path.read_text())
    labels=json.loads((data_path.parent/'evaluation.json').read_text())
    with tarfile.open(source) as package:
        manifest=json.load(package.extractfile('llm_run/run_manifest.json'))
        records=read_jsonl(package.extractfile('llm_run/'+LOG).read().decode())
    if (digest(data)!=manifest['dataset']['dataset_sha256'] or
            digest(labels)!=manifest['dataset']['evaluation_sha256']):
        raise ValueError('Dataset differs from the recorded run')
    samples={row['sample_id']:row for row in data['samples']}
    by_sample={row['sample_id']:row for row in labels['samples']}
    points,errors=[],[]
    for row in records:
        if row['task']!='T1' or row['status']!='ok':
            continue
        sample_id,model=row['sample_id'],row['model_id']
        sample=samples[sample_id]
        request,track=sample['request'],sample['track']
        horizon=json.loads(row['request'][-1]['content'])['decision']['horizon_seconds']
        now=request['time']
        try:
            traces=generate(data['map'],row['parsed'],track,now,horizon,data['rules'],
                            mode='observed_anchor_v2')
        except ValueError as error:
            errors.append(dict(sample_id=sample_id,model=model,error=str(error)))
            continue
        future=by_sample[sample_id]['future']
        if not future:
            continue
        numeric,_=local_motion_envelope(data['map'],track,now,horizon,data['rules'])
        snapshot=dict(generated_at=now,horizon=horizon,last_observed_xy=None,
                      track_id=track['id'],forecast_id=sample_id)
        for obs in future:
            dt=obs['time']-now
            if not 0.<dt<=horizon:
                continue
            actual=_compare_prediction(dict(snapshot,trajectories=traces),obs)
            control=_compare_prediction(dict(snapshot,trajectories=numeric),obs)
            if actual is None or control is None:
                continue
            band='0-1s' if dt<=1. else '1-3s' if dt<=3. else 'over-3s'
            points.append(dict(sample_id=sample_id,episode_id=request['episode_id'],model=model,
              time=obs['time'],lead=band,llm_position_error_m=actual['position_error_m'],
              numeric_position_error_m=control['position_error_m']))
    ds_keys=Counter((row['sample_id'],row['time']) for row in points if row['model'] in ('M1','M2'))
    matched=[row for row in points if row['model'] in ('M1','M2') and
             ds_keys[row['sample_id'],row['time']]==2]
    report=dict(scope='Position forecasts conditional on represented trajectory coverage. '
                      'No event calibration metrics; averages weight each forecast equally.',
                by_model_vs_numeric=summarize(points), shared_deepseek_coverage=summarize(matched),
                trajectory_errors=errors,rows=points)
    output.parent.mkdir(parents=True,exist_ok=True)
    emit(output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(args.archive,args.dataset,args.output)
