"""Episode-isolated decision orchestration over public observations only."""
from collections import Counter
import json
import math
import time

import numpy as np

from llm_formal_optimizer import horizon_for, optimize
from llm_fast_optimizer import optimize as fast_optimize
from llm_motion_client import messages_for, prompt_digest, PROMPT_VERSION
from llm_semantic import SCHEMA_VERSION, build_map, public_rules, validate_motion, normalize_motion
from llm_trajectory import generate, interpolate, local_motion_envelope
from llm_robot_rollout import prediction_horizon,selected_prediction

PUBLIC_FIELDS = {'own_xy','own_yaw','own_speed','smoke_available',
                 'remaining_seconds','action_mask','peek_count',
                 'stage_elapsed_seconds','last_action','skill_feedback'}
SAMPLE_FIELDS = {'time','xy','observer_xy','observer_yaw',
                 'uncertainty_m','stage','frame_valid'}
PROHIBITED = {'private_patrol','guard_seed','first_seed','second_seed',
              'hidden_waypoints','future_trace','referee_memory','guard_id'}


def _check_public_request(request):
    if set(request) != {'episode_id','request_id','time','stage',
                        'decision_type','public','observations','skill_state'}:
        raise ValueError('Unexpected decision request fields')
    if not all(isinstance(request[k],str) and request[k] for k in
               ('episode_id','request_id','decision_type')):
        raise ValueError('Missing request identity')
    if request['stage'] not in (0,1):
        raise ValueError('Invalid stage')
    if type(request['time']) not in (int,float) or not math.isfinite(request['time']):
        raise ValueError('Invalid global game time')
    public = request['public']
    if not isinstance(public,dict) or set(public) != PUBLIC_FIELDS:
        raise ValueError('Unexpected public state fields')
    if not isinstance(request['observations'],list) or len(request['observations'])>400:
        raise ValueError('Invalid observation batch')
    for row in request['observations']:
        if not isinstance(row,dict) or set(row) != SAMPLE_FIELDS:
            raise ValueError('Unexpected visual sample fields')
        if (type(row['time']) not in (int,float) or not math.isfinite(row['time']) or
                type(row['uncertainty_m']) not in (int,float) or
                not math.isfinite(row['uncertainty_m']) or
                not 0.<=row['uncertainty_m']<=2.):
            raise ValueError('Invalid visual timestamp or uncertainty')
        if row['stage'] not in (0,1) or row['time']>request['time']+1e-5:
            raise ValueError('Visual sample has invalid time/stage')
        if not row['frame_valid']:
            raise ValueError('Invalid camera frame cannot contain detection')
        for key in ('xy','observer_xy'):
            if (not isinstance(row[key],list) or len(row[key])!=2 or
                    not all(type(x) in (int,float) and math.isfinite(x)
                            for x in row[key])):
                raise ValueError('Invalid public coordinate')
    if (not isinstance(request['skill_state'],dict) or
            any(k in PROHIBITED for k in request['skill_state'])):
        raise ValueError('Invalid skill state')
    if any(k in PROHIBITED for k in public):
        raise ValueError('Hidden state in public request')


def _region(xy):
    x,y=xy
    found=[]
    if -4.5<=x<=-1.5 and -3.6<=y<=3.6:found.append('first')
    if 6.5<=x<=9.5 and -3.6<=y<=3.6:found.append('second')
    return found[0] if len(found)==1 else None


def _velocity(observations):
    recent=observations[-8:]
    if len(recent)<2:return [0.,0.]
    times=np.asarray([r['time'] for r in recent],float)
    coords=np.asarray([r['xy'] for r in recent],float)
    centered=times-times.mean();denom=float(centered@centered)
    if denom<1e-7:return [0.,0.]
    v=(centered@(coords-coords.mean(axis=0)))/denom
    speed=float(np.linalg.norm(v))
    return (v*min(1.,1.5/max(speed,1e-9))).tolist()


def _recent_velocity(observations):
    """Current velocity for precise 10 Hz samples, with bounded acceleration.

    A long regression lags braking/turns. Noisy or gapped measurements retain
    the smoother estimator instead of differentiating noisy positions twice.
    """
    if len(observations)<3:return _velocity(observations)
    rows=observations[-3:]
    if any(row['uncertainty_m']>.005 for row in rows):return _velocity(observations)
    dt0=rows[1]['time']-rows[0]['time'];dt1=rows[2]['time']-rows[1]['time']
    if not (.05<=dt0<=.15 and .05<=dt1<=.15):return _velocity(observations)
    old=(np.asarray(rows[1]['xy'])-rows[0]['xy'])/dt0
    latest=(np.asarray(rows[2]['xy'])-rows[1]['xy'])/dt1
    if np.linalg.norm(latest)<.03:return [0.,0.]
    acceleration=(latest-old)/(.5*(dt0+dt1))
    norm=float(np.linalg.norm(acceleration))
    acceleration*=min(1.,1.8/max(norm,1e-9))
    velocity=latest+.5*dt1*acceleration
    if velocity@latest<0.:return [0.,0.]
    speed=float(np.linalg.norm(velocity))
    return (velocity*min(1.,1.5/max(speed,1e-9))).tolist()


def _compare_prediction(snapshot, sample):
    dt=sample['time']-snapshot['generated_at']
    if not 0.<dt<=snapshot['horizon']:
        return None
    traces=[t for t in snapshot['trajectories'] if t['covered_until']+1e-6>=dt]
    if not traces:return None
    positions=np.asarray([interpolate(t['states'],dt)['xy'] for t in traces])
    weights=np.asarray([t['probability'] for t in traces],float)
    if weights.sum()<=0:return None
    weights=weights/weights.sum()
    mean=weights@positions
    observed=np.asarray(sample['xy'],float)
    distances=np.linalg.norm(positions-mean,axis=1)
    order=np.argsort(distances)
    radius=distances[order[np.searchsorted(np.cumsum(weights[order]),.8)]]
    errors=np.linalg.norm(positions-observed,axis=1)
    nearest=int(np.argmin(errors))
    realized=(traces[nearest]['hypothesis_id'] if errors[nearest]<=1.
              else 'OTHER')
    probabilities={}
    for trace in traces:
        probabilities[trace['hypothesis_id']]=(
            probabilities.get(trace['hypothesis_id'],0.)+trace['probability'])
    probabilities['OTHER']=max(0.,1.-sum(probabilities.values()))
    classes=sorted(set(probabilities)|{realized})
    brier=sum((probabilities.get(k,0.)-int(k==realized))**2 for k in classes)
    log_loss=-math.log(max(1e-9,probabilities.get(realized,0.)))
    top_class=max(probabilities,key=probabilities.get)
    direction_error=None
    if len(traces[nearest]['states'])>=2:
        pred=interpolate(traces[nearest]['states'],min(snapshot['horizon'],dt+.2))
        old=interpolate(traces[nearest]['states'],max(0.,dt-.2))
        v=np.subtract(pred['xy'],old['xy'])
        if np.linalg.norm(v)>.05 and snapshot.get('last_observed_xy') is not None:
            actual=observed-np.asarray(snapshot['last_observed_xy'])
            if np.linalg.norm(actual)>.05:
                direction_error=abs(math.atan2(v[1],v[0])-
                                    math.atan2(actual[1],actual[0]))
                direction_error=min(direction_error,2*math.pi-direction_error)
    return dict(time=sample['time'],track_id=snapshot['track_id'],
                forecast_id=snapshot['forecast_id'],
                position_error_m=float(np.linalg.norm(mean-observed)),
                coverage80=bool(np.linalg.norm(mean-observed)<=
                                radius+sample['uncertainty_m']+1e-6),
                direction_error_rad=direction_error,
                event_proxy_class=realized,brier_proxy=float(brier),
                log_loss_proxy=float(log_loss),
                predicted_top_probability=float(probabilities[top_class]),
                top_class_correct=bool(top_class==realized),
                note='Event class is a nearest-trajectory proxy, not ground truth patrol target')


class EpisodeContext:
    def __init__(self, episode_id, map_id, rules_version, velocity_mode='legacy'):
        self.episode_id=episode_id
        self.map_id=map_id
        self.rules_version=rules_version
        self.velocity_mode=velocity_mode
        self.tracks={}
        self.track_counter=0
        self.requests=[]
        self.predictions=[]
        self.comparisons=[]
        self.plans=[]
        self.llm_messages={}
        self.llm_attempts={}
        self.responses={}
        self.last_time=-1.
        self.stage=0
        self.status='active'

    def add_samples(self, samples):
        for sample in samples:
            region=_region(sample['xy'])
            if region is None:
                continue
            candidates=[t for t in self.tracks.values() if t['region']==region]
            compatible=[]
            for track in candidates:
                previous=track['observations'][-1]
                dt=sample['time']-previous['time']
                if (dt>=0. and math.dist(sample['xy'],previous['xy']) <=
                        1.5*dt+sample['uncertainty_m']+previous['uncertainty_m']+.35):
                    compatible.append(track)
            if len(compatible)==1:
                track=compatible[0]
            elif not candidates:
                self.track_counter+=1
                track_id=f'track-{self.track_counter:04d}'
                track=dict(id=track_id,region=region,observations=[],
                           association_uncertainty=0.,estimated_velocity=[0.,0.],
                           hypothesis_history=[])
                self.tracks[track_id]=track
            else:
                # Ambiguous or kinematically inconsistent: retain uncertainty
                # rather than stealing an established guard's identity.
                for item in candidates:
                    item['association_uncertainty']=1.
                continue
            track['observations'].append(sample)
            track['estimated_velocity']=(_recent_velocity(track['observations'])
                if self.velocity_mode=='recent_acceleration_v2' else _velocity(track['observations']))
            for snapshot in self.predictions:
                if snapshot['track_id']==track['id'] and sample['time']>snapshot['generated_at']:
                    comparison=_compare_prediction(snapshot,sample)
                    if comparison is not None:self.comparisons.append(comparison)

    def track_for(self, stage):
        region='second' if stage else 'first'
        found=[t for t in self.tracks.values() if t['region']==region and
               t['association_uncertainty']<1.]
        return found[0] if len(found)==1 else None

    def evaluation(self,summary=None):
        rows=self.comparisons
        direction=[x['direction_error_rad'] for x in rows
                   if x['direction_error_rad'] is not None]
        reliability=[]
        for index,lo in enumerate((0.,.2,.4,.6,.8)):
            bucket=[x for x in rows if min(4,int((
                x['predicted_top_probability']+1e-12)*5))==index]
            reliability.append(dict(bin=f'{lo:.1f}-{lo+.2:.1f}',count=len(bucket),
                mean_prediction=(sum(x['predicted_top_probability'] for x in bucket)/len(bucket)
                                 if bucket else None),
                empirical_accuracy=(sum(x['top_class_correct'] for x in bucket)/len(bucket)
                                    if bucket else None)))
        action_scores=[]
        if summary is not None:
            for request in self.requests:
                response=self.responses.get(request['request_id'],{})
                if response.get('status') not in ('plan','boundary'):continue
                risk=response['optimizer']['selected_risk']
                failed=(summary['handoff_time'] is None if request['stage']==0
                        else summary['winner']!='attacker')
                action_scores.append(dict(stage=request['stage'],
                    predicted_failure=risk,actual_failure=bool(failed),
                    brier=(risk-float(failed))**2))
        return dict(samples=len(rows),
            mean_position_error_m=(sum(x['position_error_m'] for x in rows)/len(rows)
                                   if rows else None),
            coverage80=(sum(x['coverage80'] for x in rows)/len(rows) if rows else None),
            mean_direction_error_rad=(sum(direction)/len(direction)
                                      if direction else None),
            route_event_proxy_accuracy=(sum(x['top_class_correct'] for x in rows)/len(rows)
                                        if rows else None),
            mean_brier_proxy=(sum(x['brier_proxy'] for x in rows)/len(rows)
                              if rows else None),
            mean_log_loss_proxy=(sum(x['log_loss_proxy'] for x in rows)/len(rows)
                                if rows else None),
            reliability_proxy=reliability,
            action_risk_scores=action_scores,
            mean_action_risk_brier=(sum(x['brier'] for x in action_scores)/len(action_scores)
                                    if action_scores else None))


class DecisionOrchestrator:
    def __init__(self, reasoner, config):
        self.map=build_map()
        self.rules=public_rules()
        self.reasoner=reasoner
        self.config=config
        self.episodes={}

    def _observe_action(self, request, track, reason):
        mask=request['public']['action_mask']
        if not mask[2] and not mask[3]:
            raise ValueError('Observation requested but no peek action is legal')
        first_preferred=(request['public']['peek_count']==0 or
                         (track and track['observations'][-1]['xy'][1]<0.))
        action=2 if first_preferred and mask[2] else 3 if mask[3] else 2
        return dict(status='observe',action=action,
                    gate=dict(sufficient=False,reasons=[reason],
                              peek_count=request['public']['peek_count']))

    def _can_repeek(self, request):
        public=request['public']
        return (public['peek_count']<self.config.get('max_peeks_per_stage',3)
                and request['skill_state'].get('observation_window_elapsed',
                                               public['stage_elapsed_seconds'])<
                    self.config.get('max_observation_seconds',12.)
                and (public['action_mask'][2] or public['action_mask'][3]))

    def _boundary_decision(self,request,context,track):
        started=time.perf_counter()
        public=dict(request['public'],retreat_allowed=request['skill_state'].get('retreats',0)<1)
        if track is None or len(track['observations'])<2:
            raise ValueError('Boundary reassessment has no public guard track')
        horizon=prediction_horizon(public,request['stage'])
        trajectories,unknown=local_motion_envelope(
            self.map,track,request['time'],horizon,self.rules)
        plan=fast_optimize(trajectories,request['stage'],public['smoke_available'],
            unknown,horizon,self.map,public['own_xy'],
            maximum_particles=self.config.get('maximum_fast_particles',18),own_state=public)
        waited=request['skill_state'].get('boundary_waits',0)
        eligible=[row for row in plan['candidates'] if row['wait']<=max(0.,2.-waited)]
        best=min(eligible,key=lambda row:(row['objective'],row['wait']))
        plan['selected']={key:best[key] for key in ('side','wait','smoke','maneuver')}
        plan['selected_risk']=best['risk']
        plan['robot_prediction']=selected_prediction(public,request['stage'],best)
        prediction=dict(forecast_id=f"{request['episode_id']}:{request['request_id']}",
            track_id=track['id'],generated_at=request['time'],horizon=horizon,
            hypotheses=[],trajectories=trajectories,last_observed_xy=track['observations'][-1]['xy'],
            forecast_source='local_motion_envelope_v1')
        context.predictions.append(prediction)
        return self._record(context,request['request_id'],dict(status='boundary',
            action=1 if best['maneuver']=='retreat' else 4 if best['wait']>0. else 7,
            optimizer=plan,forecast=prediction,
            gate=dict(sufficient=True,reasons=['new_visual_boundary_reassessment']),
            timing=dict(api_seconds=0.,optimizer_seconds=plan['score_seconds'],
                        local_boundary_seconds=time.perf_counter()-started),
            llm_called=False))

    def decide(self, request):
        _check_public_request(request)
        episode_id=request['episode_id']
        context=self.episodes.setdefault(episode_id,EpisodeContext(
            episode_id,self.map['map_id'],self.rules['version'],
            velocity_mode=self.config.get('velocity_mode','legacy')))
        req_id=request['request_id']
        if req_id in context.responses:
            original=next(row for row in context.requests
                          if row['request_id']==req_id)
            if original!=request:
                raise ValueError('Duplicate Request ID has different public payload')
            return context.responses[req_id]
        if context.status!='active' or request['time']+1e-7<context.last_time:
            raise ValueError('Episode closed or game time moved backward')
        context.last_time=request['time'];context.stage=request['stage']
        context.add_samples(request['observations'])
        context.requests.append(request)
        track=context.track_for(request['stage'])
        public=request['public']
        if public['action_mask'][7] and self.config.get('boundary_recheck',False):
            return self._boundary_decision(request,context,track)
        if public['peek_count']==0:
            result=self._observe_action(request,track,'initial_observation')
            return self._record(context,req_id,result)
        if (public['peek_count']<self.config.get('minimum_peeks_per_stage',1)
                and self._can_repeek(request)):
            return self._record(context,req_id,
                self._observe_action(request,track,'configured_observation_budget'))
        if track is None or len(track['observations'])<2:
            if self._can_repeek(request):
                result=self._observe_action(request,track,'insufficient_legal_track')
                return self._record(context,req_id,result)
            raise ValueError('No associated two-sample guard track after observation cap')
        last=track['observations'][-1]
        if request['time']-last['time']>1.5 and self._can_repeek(request):
            result=self._observe_action(request,track,'stale_visual_track')
            return self._record(context,req_id,result)
        budget=self.config.get('max_llm_calls_per_stage')
        used=sum(row['request_id'] in context.llm_messages and row['stage']==request['stage']
                 for row in context.requests)
        local_update=(budget is not None and used>=budget)
        if (local_update and math.hypot(*track['estimated_velocity'])<=.15
                and self._can_repeek(request)):
            result=self._observe_action(request,track,'stationary_local_motion_is_ambiguous')
            result['llm_called']=False
            result['timing']=dict(api_seconds=0.,preparation_seconds=0.,optimizer_seconds=0.)
            return self._record(context,req_id,result)
        prepared_at=time.perf_counter()
        stateful=self.config.get('robot_rollout_mode')=='stateful_v2'
        horizon=(prediction_horizon(public,request['stage']) if stateful else
                 horizon_for(request['stage'],public['own_xy'],public['remaining_seconds']))
        preparation_seconds=time.perf_counter()-prepared_at
        recent=track['observations'][-30:]
        prior=[dict(hypotheses=x['hypotheses'],generated_at=x['generated_at'])
               for x in context.predictions if x['track_id']==track['id']][-2:]
        comparisons=[x for x in context.comparisons if x['track_id']==track['id']][-8:]
        reasoning_request=dict(schema=SCHEMA_VERSION,
            request_id=req_id,map_id=self.map['map_id'],
            track_id=track['id'],region=track['region'],
            game_time=request['time'],horizon_seconds=horizon,
            latest_measurement_at=last['time'],
            measurement_age_seconds=request['time']-last['time'],
            path_coverage_seconds=horizon+request['time']-last['time'],
            current_robot_state=public,skill_state=request['skill_state'],
            recent_observations=recent,
            older_observation_count=max(0,len(track['observations'])-len(recent)),
            estimated_guard_velocity=track['estimated_velocity'],
            observable_guard_facing=None,
            association_uncertainty=track['association_uncertainty'],
            previous_hypotheses=prior,prediction_comparisons=comparisons)
        local_started=time.perf_counter();precomputed=None
        if local_update:
            precomputed,unknown=local_motion_envelope(
                self.map,track,request['time'],horizon,self.rules)
            parsed=dict(schema=SCHEMA_VERSION,hypotheses=[])
            inferred=None;messages=None;llm_seconds=0.
        else:
            messages=messages_for(reasoning_request,self.map,self.rules)
            context.llm_messages[req_id]=messages
            wall_start=time.perf_counter()
            inferred=self.reasoner.infer(messages)
            context.llm_attempts[req_id]=inferred
            parsed=normalize_motion(json.loads(inferred['text']),self.map,track['region'])
            validate_motion(parsed,self.map,track['id'],track['region'],horizon)
            llm_seconds=time.perf_counter()-wall_start
            unknown=sum(h['probability'] for h in parsed['hypotheses']
                        if h['state']=='UNKNOWN')
        probs=[h['probability'] for h in parsed['hypotheses'] if h['probability']>0]
        entropy=-sum(p*math.log(p) for p in probs)/max(1e-9,math.log(max(2,len(probs))))
        entropy_gate=(self.config.get('observation_gate_mode','legacy_entropy')
                      =='legacy_entropy' and entropy>.92)
        if (unknown>.45 or entropy_gate) and self._can_repeek(request):
            result=self._observe_action(request,track,
                'high_unknown_mass' if unknown>.45 else 'diffuse_motion_hypotheses')
            if inferred is not None:
                result['reasoning']=dict(raw_request=messages,raw_response=inferred,
                    hypotheses=parsed,api_seconds=llm_seconds,prompt_version=PROMPT_VERSION)
            return self._record(context,req_id,result)
        model_started=time.perf_counter()
        trajectories=(precomputed if precomputed is not None else
                      generate(self.map,parsed,track,request['time'],horizon,self.rules,
                               mode=self.config.get('trajectory_mode','legacy')))
        model_seconds=(time.perf_counter()-local_started if local_update else
                       time.perf_counter()-model_started)
        prediction=dict(forecast_id=f'{episode_id}:{req_id}',
            track_id=track['id'],generated_at=request['time'],
            horizon=horizon,hypotheses=parsed['hypotheses'],
            trajectories=trajectories,
            last_observed_xy=last['xy'],
            forecast_source='local_motion_envelope_v1' if local_update else 'llm_paths')
        context.predictions.append(prediction)
        track['hypothesis_history'].append(dict(time=request['time'],
                                                 hypotheses=parsed['hypotheses']))
        mode=self.config.get('optimizer_mode','formal')
        if mode=='fast_proxy':
            plan=fast_optimize(trajectories,request['stage'],
                public['smoke_available'],unknown,horizon,self.map,
                public['own_xy'],
                maximum_particles=self.config.get('maximum_fast_particles',
                    self.config.get('maximum_formal_particles',6)),
                own_state=public if stateful else None)
        elif mode=='formal':
            plan=optimize(trajectories,request['stage'],
                public['smoke_available'],unknown,horizon,
                maximum_particles=self.config.get('maximum_formal_particles',6))
        else:
            raise ValueError(f'Unknown optimizer_mode: {mode}')
        sorted_actions=sorted(plan['candidates'],key=lambda row:row['objective'])
        margin=(sorted_actions[1]['objective']-sorted_actions[0]['objective']
                if len(sorted_actions)>1 else 1.)
        by_hyp={}
        for candidate in plan['candidates']:
            for outcome in candidate['outcomes']:
                key=outcome['hypothesis_id']
                score=by_hyp.setdefault(key,{})
                action=(candidate['side'],candidate['wait'],candidate['smoke'])
                score[action]=score.get(action,0.)+outcome['weight']*outcome['survived']
        best_by_hyp=[max(scores,key=scores.get) for scores in by_hyp.values()]
        disagreement=(len(set(best_by_hyp))>1)
        gate=dict(sufficient=True,reasons=[],entropy=entropy,
                  unknown_probability=unknown,action_margin=margin,
                  hypothesis_action_disagreement=disagreement,
                  peek_count=public['peek_count'])
        if self._can_repeek(request) and (
                (disagreement and margin<.12) or
                (margin<.025 and plan['selected_risk']>.4)):
            result=self._observe_action(request,track,'action_disagreement_or_close_scores')
            result['gate'].update(gate,sufficient=False,
                                  reasons=['action_disagreement_or_close_scores'])
            result['optimizer']=plan
        else:
            result=dict(status='plan',plan=plan['selected'],gate=gate,
                        boundary_recheck=self.config.get('boundary_recheck',False),
                        optimizer=plan,track_id=track['id'],
                        forecast_id=prediction['forecast_id'])
            context.plans.append(dict(time=request['time'],request_id=req_id,
                                      plan=plan['selected']))
        if inferred is not None:
            result['reasoning']=dict(raw_request=messages,raw_response=inferred,
                hypotheses=parsed,api_seconds=llm_seconds,prompt_version=PROMPT_VERSION,
                prompt_digest=prompt_digest())
        result['llm_called']=inferred is not None
        result['forecast']=prediction
        result['timing']=dict(api_seconds=llm_seconds,
                              preparation_seconds=preparation_seconds,
                              trajectory_seconds=model_seconds,
                              optimizer_seconds=plan['score_seconds'])
        if stateful:
            result['time_alignment']=dict(clock='simulation_game_time',
                request_game_time=request['time'],forecast_generated_at=request['time'],
                latest_measurement_at=last['time'],
                measurement_age_seconds=request['time']-last['time'],
                execution_mode='simulation_paused_while_waiting_for_api')
        return self._record(context,req_id,result)

    def _record(self,context,request_id,result):
        response=dict(episode_id=context.episode_id,request_id=request_id,
                      game_time=context.last_time,map_id=context.map_id,
                      rules_version=context.rules_version,
                      status=result['status'],**{k:v for k,v in result.items()
                                                if k!='status'})
        context.responses[request_id]=response
        return response

    def finish(self,episode_id,summary,observations=()):
        context=self.episodes[episode_id]
        context.add_samples(observations)
        context.status='finished'
        return dict(episode_id=episode_id,summary=summary,
                    evaluation=context.evaluation(summary),
                    forecast_comparisons=context.comparisons,
                    track_count=len(context.tracks),
                    request_count=len(context.requests),
                    plan_count=len(context.plans))
