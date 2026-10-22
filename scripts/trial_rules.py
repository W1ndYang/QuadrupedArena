"""CPU-only R0 virtual-mark referee. No robot actuation or projectile physics."""
import copy
import math
import numpy as np
from tactical_logic import block, wrap


def cumulative_hazard(age, maximum, tau):
    # expm1 avoids cancellation at small ages.
    return maximum * (age + tau * math.expm1(-age / tau))


def covers_at(config, t):
    result = copy.deepcopy(config["arena"]["covers"])
    if t >= config["collapse"]["time"]:
        result = [copy.deepcopy(config["collapse"]["debris"]) if c["name"] == config["collapse"]["wall"] else c for c in result]
    return result


def aim_valid(config, xy, yaw, i, covers):
    j = 1-i
    d = np.asarray(xy[j])-xy[i]
    if np.linalg.norm(d) > config["range_m"] or np.linalg.norm(d) < 1e-8:
        return False
    if abs(wrap(math.atan2(d[1],d[0])-yaw[i])) > math.radians(config["aim_half_angle_degrees"])+1e-10:
        return False
    h=config["representative_heights"]
    return block((*xy[i],h[i]),(*xy[j],h[j]),covers) is None


class Referee:
    """End-of-step events; elapsed legal score is settled before boundary hits.

    Geometry is sampled at fixed dt. Boundary crossing and hit timing are thus
    quantized to that dt; the hazard law itself is independent of render FPS.
    Each aim episode draws one exponential threshold from an actor-local stream.
    """
    def __init__(self,config):
        self.cfg=config; self.hp=[config["initial_hp"]]*2
        n=len(config["arena"]["task_points"])
        self.eligible=[True]*n; self.dwell=[0.]*n; self.scores=[0.]*n
        self.aim=[0.,0.]; self.threshold=[None,None]; self.cooldown_until=[0.,0.]
        self.rng=[np.random.default_rng(s) for s in np.random.SeedSequence(config["seed"]).spawn(2)]
        self.inside=[False]*n; self.last_valid=[False,False]
        self.last_t=None; self.events=[]; self.finished=None; self.scoring=[False]*n
        self.rotation=config.get('target_rotation',{})
        self.rotating=self.rotation.get('enabled',False)
        self.active_target=None;self.generation=0;self.triggered_at=None;self.expires_at=None
        self.instance_score=0.
        if self.rotating:
            if n<2:raise ValueError('Rotating targets require at least two locations')
            if not math.isfinite(self.rotation['lifetime_seconds']) or self.rotation['lifetime_seconds']<=0:
                raise ValueError('Target lifetime must be finite and positive')
            self.eligible=[False]*n
            self.activate_target(0.,0)

    def activate_target(self,t,index):
        self.active_target=index;self.generation+=1
        self.eligible=[k==index for k in range(len(self.eligible))]
        self.dwell=[0.]*len(self.dwell);self.scoring=[False]*len(self.scoring)
        self.triggered_at=None;self.expires_at=None;self.instance_score=0.
        self.emit(t,'target_spawned',point=index,generation=self.generation)

    def retire_target(self,t,reason):
        k=self.active_target;self.eligible[k]=False;self.dwell[k]=0.;self.scoring[k]=False
        self.emit(t,'target_invalidated',point=k,generation=self.generation,reason=reason,
                  triggered_at=self.triggered_at,expires_at=self.expires_at,instance_score=self.instance_score)

    def emit(self,t,kind,**fields):
        self.events.append(dict(time=round(t,6),type=kind,**fields))

    def reset_aim(self,i):
        self.aim[i]=0.; self.threshold[i]=None; self.last_valid[i]=False

    def step(self,t,xy,yaw,covers,map_changed=False,attack_enabled=None):
        if self.finished: raise RuntimeError("Referee already finished")
        if self.last_t is not None and t < self.last_t: raise ValueError("Clock reversed")
        dt=0. if self.last_t is None else min(t,self.cfg["duration"])-self.last_t
        inside=[float(np.linalg.norm(np.asarray(xy[0])-p["center"])) <= p["radius"]+1e-10
                for p in self.cfg["arena"]["task_points"]]
        if sum(inside)>1: raise ValueError("Overlapping target membership")
        self.scoring=[False]*len(inside)
        advance=False
        # The elapsed interval uses the last sample's occupancy. Geometry/time
        # quantization is documented; no retroactive deletion on the hit boundary.
        for k,was_inside in enumerate(self.inside):
            if was_inside and self.eligible[k] and self.hp[0]>0:
                elapsed=dt
                if self.rotating and self.expires_at is not None:
                    elapsed=max(0.,min(t,self.cfg['duration'],self.expires_at)-(self.last_t or 0.))
                old=self.dwell[k]; new=old+elapsed
                gain=self.cfg["score_rate"]*(max(0.,new-self.cfg["hold_seconds"])-max(0.,old-self.cfg["hold_seconds"]))
                self.dwell[k]=new;self.scores[k]+=gain;self.scoring[k]=gain>0
                if self.rotating:self.instance_score+=gain
                if gain>0 and old<=self.cfg["hold_seconds"]:
                    self.emit(t,"scoring_started",point=k)
            else: self.dwell[k]=0.
        for k,(before,now) in enumerate(zip(self.inside,inside)):
            if before!=now:
                if not self.rotating or self.eligible[k]:
                    self.emit(t,"target_entered" if now else "target_left",point=k)
                self.dwell[k]=0.;self.scoring[k]=False
        self.inside=inside
        if t>=self.cfg["duration"]-1e-9:
            self.finished="time_limit";self.last_t=t
            self.emit(t,"finish",reason=self.finished,score=self.score)
            return
        if self.rotating and self.expires_at is not None and t>=self.expires_at-1e-9:
            self.retire_target(self.expires_at,'timeout');advance=True
        if map_changed:
            # Map changes trigger normal LOS re-evaluation below, not a global
            # aim reset: an unaffected continuous line of sight remains valid.
            self.emit(t,"collapse",wall=self.cfg["collapse"]["wall"],angle=self.cfg["collapse"]["angle_degrees"])
        hits=[]
        for i in range(2):
            valid=(attack_enabled is None or attack_enabled[i]) and min(self.hp)>0 and t>=self.cooldown_until[i]-1e-9 and aim_valid(self.cfg,xy,yaw,i,covers)
            if not valid:
                if self.aim[i]>0:self.emit(t,"aim_broken",actor=i)
                self.reset_aim(i);continue
            if not self.last_valid[i]:
                self.threshold[i]=float(self.rng[i].exponential())
                self.emit(t,"aim_started",actor=i,threshold=self.threshold[i])
                elapsed=0.
            else:elapsed=dt
            old=self.aim[i];self.aim[i]+=elapsed;self.last_valid[i]=True
            hazard=cumulative_hazard(self.aim[i],self.cfg["lambda_max"],self.cfg["tau"][i])
            if hazard+1e-12>=self.threshold[i]:
                hits.append((i,self.aim[i],self.threshold[i],hazard))
        # All candidates were generated from the same pre-damage HP snapshot.
        for i,age,threshold,hazard in hits:
            self.hp[1-i]=max(0,self.hp[1-i]-1)
            self.cooldown_until[i]=t+self.cfg["cooldown_seconds"]
            self.emit(t,"hit",attacker=i,target=1-i,aim_seconds=age,threshold=threshold,
                      hazard=hazard,cooldown_until=self.cooldown_until[i],score_before=self.score)
            self.reset_aim(i)
            if i==1:
                for k,now in enumerate(inside):
                    if now and self.eligible[k]:
                        if self.rotating:
                            self.retire_target(t,'hit');advance=True
                        else:
                            self.eligible[k]=False;self.dwell[k]=0.;self.scoring[k]=False
                            self.emit(t,"target_invalidated",point=k)
        for i in range(2):
            if self.hp[i]==0:
                self.reset_aim(i)
                if any(hit[0]==1-i for hit in hits):self.emit(t,"eliminated",actor=i)
        if min(self.hp)==0:
            for i in range(2):self.reset_aim(i)
        reasons=[]
        if self.hp[0]==0:reasons.append("quadruped_eliminated")
        if self.rotating:
            if advance and self.hp[0]>0:
                self.activate_target(t,(self.active_target+1)%len(self.eligible))
            if self.hp[0]>0 and self.eligible[self.active_target] and inside[self.active_target] and self.triggered_at is None:
                self.triggered_at=t;self.expires_at=t+self.rotation['lifetime_seconds']
                self.emit(t,'target_triggered',point=self.active_target,generation=self.generation,expires_at=self.expires_at)
        elif not any(self.eligible):reasons.append("all_targets_invalid")
        if reasons:
            self.finished=reasons[0]
            self.emit(t,"finish",reason=self.finished,concurrent_reasons=reasons,score=self.score)
        self.last_t=t

    @property
    def score(self):return round(sum(self.scores),6)

    def snapshot(self,t):
        result=dict(hp=self.hp.copy(),eligible=self.eligible.copy(),dwell=self.dwell.copy(),
            score=self.score,target_scores=self.scores.copy(),aim=self.aim.copy(),
            rates=[self.cfg["lambda_max"]*(-math.expm1(-self.aim[i]/self.cfg["tau"][i])) for i in range(2)],
            cooldown=[max(0.,u-t) for u in self.cooldown_until],inside=self.inside.copy(),
            scoring=self.scoring.copy(),finished=self.finished)
        if self.rotating:
            result.update(active_target=self.active_target,target_generation=self.generation,
                target_triggered_at=self.triggered_at,target_expires_at=self.expires_at,
                target_remaining=None if self.expires_at is None else round(max(0.,self.expires_at-t),6),
                target_instance_score=round(self.instance_score,6))
        return result
