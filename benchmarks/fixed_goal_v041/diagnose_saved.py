"""No API. Replay accepted actions only while every public input matches exactly."""
import copy,math,tempfile
from pathlib import Path
from ds_fixed_goal import R,scenario,read,dump,digest
from peek_braking_fix import SequenceProbe
from tactical_logic import wrap
from trial_motion import clearance

class Instrumented(SequenceProbe):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.proposed=[]
        # Observe movement proposals, without changing the step or collision response.
        b=self.actor[1];original=b.step
        def step(*args,**kwargs):
            before=copy.deepcopy(b.__dict__);original(*args,**kwargs)
            if not self.planner.segment(before['xy'],b.xy):
                self.proposed.append(dict(time=self.trace[-1]['t'],before=before['xy'].tolist(),proposed=b.xy.tolist(),
                    speed_before=before['v'],proposed_speed=b.v,travel_yaw=before['travel_yaw'],
                    route=[v.tolist() for v in before['route']],
                    violating_walls=[dict(name=c['name'],clearance_m=clearance(b.xy,c)) for c in self.planner.obstacles if clearance(b.xy,c)<self.planner.margin]))
        b.step=step

def main():
    c=read(R/'cases.json')[0];scenario(c['layout']);f=R/'run_1';cfg=read(f/'trial_config.json');count=0
    def stored(packet):
        nonlocal count
        i=packet['observation_id'];request=read(f/f'api_calls/{i:03d}_request.json')
        assert digest(packet)==digest(request['observation']),('public input diverged',i)
        count+=1;return read(f/f'accepted/{i:03d}.json')['action']
    with tempfile.TemporaryDirectory() as td:
        p=Instrumented(cfg['base'],cfg['rules'],stored,Path(td));result=p.run('diagnostic_same_actions')
    assert result['states']==read(f/'timeline.json'),'Instrumented replay changed trajectory'
    assert result['events']==read(f/'events.json'),'Instrumented replay changed events'
    record=dict(status='passed',model_calls=0,public_packets_matched=count,trajectory_identical=True,
        human_rejected_step_count=len(p.proposed),proposals=p.proposed,
        interpretation='Wall clearance rejects proposed humanoid steps; the mover remains at its old position with speed reset. This diagnostic does not repair the mover or measure a corrected-policy outcome.')
    dump(f/'movement_diagnostic.json',record)
    print('DIAGNOSTIC',count,'matching packets;',len(p.proposed),'rejected steps;',p.proposed[-1] if p.proposed else '')

if __name__=='__main__':main()
