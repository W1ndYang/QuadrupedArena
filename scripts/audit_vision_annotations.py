"""Post-terminal only: compare submitted ground projections with referee centers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from llm_trial import dump
from semantic_skills import ground_point


def audit(root):
    read=lambda name:json.loads((root/name).read_text())
    result=read('result.json')
    assert result['status']=='passed' and result.get('finish'), 'Terminal matches only'
    states={round(s['t'],5):s for s in read('timeline.json')}
    rows=[]
    for d in read('decisions.json'):
        sight=d['params'].get('sighting')
        if not sight:continue
        obs=read(f"observations/{d['observation_id']:03d}.json")
        frame=next(f for f in obs['frames'] if f['simulation_seconds']==sight['time'])
        projected=ground_point(frame,sight['feet'])
        actual=states[round(sight['time'],5)]['xy'][1]
        rows.append(dict(decision=d['observation_id'],frame=frame['file'],time=sight['time'],
            feet=sight['feet'],projected_xy=projected,posthoc_referee_center_xy=actual,
            center_error_m=math.dist(projected,actual),
            frame_sha256=hashlib.sha256((root/frame['file']).read_bytes()).hexdigest()))
    value=dict(written_after_match=True,not_planner_input=True,annotations=rows,
        caveat='Small selected sample only. Euclidean distance to actor center at image time, not pixel ground-truth accuracy. Includes calibration/ground-plane and gait offsets. Does not establish causation or overall perception quality.')
    dump(root/'annotation_audit.json',value)
    print(json.dumps(dict(run=str(root),errors_m=[row['center_error_m'] for row in rows])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path)
    for root in p.parse_args().runs:audit(root)
