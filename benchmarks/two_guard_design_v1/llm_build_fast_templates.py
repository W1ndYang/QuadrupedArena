"""Build robot-only action traces with the formal referee, before live runs."""
import json
from pathlib import Path

from formal_central_screen import run
from llm_semantic import build_map


HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'llm_fast_templates.json'


def build():
    rows=[]
    for stage in (0,1):
        # Put the dummy guard in the other region so it cannot change the
        # robot-only path through physical contact.
        x=-3. if stage else 8.
        guard=dict(start_xy=[x,3.],start_yaw=-1.57,
                   initial_speed=0.,speed_mps=1.,
                   waypoints=[[x,-3.]],dwell_node_indices=[],
                   dwell_seconds=0.)
        for side in ('north','south'):
            for wait in (0.,1.,2.):
                for smoke in (False,True):
                    result=run(0,side,wait,14.,0.,7.1 if stage else -3.9,
                               4.,.6,True,second_peek=False,skip_peek=True,
                               route_y=1.6,tag_seconds=.6,peek_if_unseen=False,
                               stage2_local=bool(stage),patrol_style='hypothesis',
                               hypothesis_guard=guard,use_smoke=smoke)
                    if not result['local_survived']:
                        raise RuntimeError(f'Template did not reach goal: {stage,side,wait,smoke}')
                    rows.append(dict(stage=stage,side=side,wait=wait,
                                     smoke=smoke,seconds=result['seconds'],
                                     attacker_trace=result['attacker_trace']))
                    print(f'Built {stage=} {side=} {wait=} {smoke=}',flush=True)
    return dict(schema='llm-fast-action-templates-v1',
                map_id=build_map()['map_id'],rows=rows)


if __name__=='__main__':
    OUTPUT.write_text(json.dumps(build(),separators=(',',':'))+'\n')
    print(OUTPUT)
