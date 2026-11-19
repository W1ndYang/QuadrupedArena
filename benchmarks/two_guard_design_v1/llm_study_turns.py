"""Audit whether legal observations can support observable stop-and-turn labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from llm_semantic import build_map, public_rules
from llm_study_data import (archived_logs, diagnostic_logs, emit, episode_candidates,
                            motion_messages, read_jsonl)


SCHEMA = 'public-stop-and-redirect-v1'
MIN_MOVING_SPEED = .25
MAX_STOP_DRIFT = .03
OBSERVATION_PERIOD = .1
TIME_TOLERANCE = .005


def observation_runs(samples):
    """Return precise continuous 10 Hz runs within a single public region."""
    ordered = {(row['stage'], row['time']): row for row in samples
               if row['frame_valid'] and row['uncertainty_m'] <= .005}
    runs = []
    for _, row in sorted(ordered.items()):
        if not runs or row['stage'] != runs[-1][-1]['stage'] or abs(
                row['time']-runs[-1][-1]['time']-OBSERVATION_PERIOD) > TIME_TOLERANCE:
            runs.append([])
        runs[-1].append(row)
    return runs


def direction_change(points):
    """Identify a completed stop followed by a changed translation direction.

    Require 0.3 s of motion on each side and a measured 0.2 s near-stationary
    interval between them. A new departure must be visible within 1.4 s of
    the incoming motion. Timestamp bounds delimit the stationary observation,
    not an unobserved torso turn or private controller-state transition.
    """
    events = []
    used_until = -1
    for i in range(3,len(points)-8):
        if i-3 <= used_until:
            continue
        t_in, x_in = points[i]
        t_left, x_left = points[i-3]
        before = [(a-b)/(t_in-t_left) for a,b in zip(x_in,x_left)]
        speed_before = math.hypot(*before)
        if speed_before < MIN_MOVING_SPEED:
            continue
        for j in range(i+5,min(len(points)-3,i+15)):
            stop = next((k for k in range(i,j-1)
                         if max(math.dist(points[k][1], points[q][1])
                                for q in (k+1,k+2)) <= MAX_STOP_DRIFT), None)
            if stop is None:
                continue
            t_right, x_right = points[j+3]
            t_out, x_out = points[j]
            after = [(a-b)/(t_right-t_out) for a,b in zip(x_right,x_out)]
            speed_after = math.hypot(*after)
            if speed_after < MIN_MOVING_SPEED:
                continue
            cosine = sum(a*b for a,b in zip(before,after))/(speed_before*speed_after)
            angle = math.degrees(math.acos(min(1.,max(-1.,cosine))))
            if angle < 45.:
                continue
            events.append(dict(event='REVERSE' if angle >= 150. else 'TURN',
                               heading_change_degrees=angle, incoming_speed=speed_before,
                               outgoing_speed=speed_after, evidence_from=t_left,
                               last_inbound_at=t_in,
                               stationary_interval=[points[stop][0], points[stop+2][0]],
                               first_outbound_at=t_out,evidence_to=t_right))
            used_until = j+3
            break
    return events


def audit_case(samples):
    runs = observation_runs(samples)
    stages = defaultdict(list)
    for run in runs:
        stages[run[0]['stage']].append(run)
    results = []
    for stage,runs_in_stage in sorted(stages.items()):
        for run in runs_in_stage:
            if len(run) >= 13:
                events = direction_change([(row['time'],row['xy']) for row in run])
            else:
                events = []
            results.append(dict(stage=stage,start=run[0]['time'],end=run[-1]['time'],
                                frames=len(run),can_fit_detector=len(run)>=13,events=events))
    return results


def observations(events):
    for event in events:
        if event['kind'] == 'decision':
            yield from event['request']['observations']
        elif event['kind'] == 'finish':
            yield from event['payload']['observations']


def historical_cases(archive, diagnostics):
    for log,provenance in archived_logs(archive):
        yield 'archived_numeric',provenance,list(observations(log)),log
    if diagnostics:
        for log,provenance in diagnostic_logs(diagnostics):
            yield 'historical_diagnostic',provenance,list(observations(log)),log


def supplemental_cases(directory):
    report = json.loads((directory/'collector_manifest.json').read_text())
    if not report.get('complete'):
        raise ValueError('Use a complete supplemental collection')
    for row in report['episodes']:
        observations_path = directory/row['log']
        contents = observations_path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != row['log_sha256']:
            raise ValueError('Supplemental public observations differ from their frozen hash')
        decisions = (directory/row['decision_log']).read_bytes()
        if hashlib.sha256(decisions).hexdigest() != row['decision_sha256']:
            raise ValueError('Supplemental public decisions differ from their frozen hash')
        request_log = [dict(event,kind='decision') for event in json.loads(decisions)['requests']]
        yield 'new_fixed_probe',row,read_jsonl(contents.decode()),request_log


def forecast_anchors(runs, request_log, provenance, semantic_map, rules):
    events = [(run['stage'], event) for run in runs for event in run['events']]
    if not events:
        return []
    candidates = episode_candidates(request_log, provenance, semantic_map, rules,
                                   'observation-coverage-audit')
    rows = []
    for stage,event in events:
        eligible = []
        for sample in candidates:
            request = sample['request']
            if request['stage'] != stage or request['time'] > event['evidence_from']:
                continue
            exchange = motion_messages(sample,semantic_map,rules)
            if exchange is None:
                continue
            horizon = json.loads(exchange[-1]['content'])['decision']['horizon_seconds']
            if request['time']+horizon+1e-6 >= event['evidence_to']:
                eligible.append(request['time'])
        rows.append(dict(event=event['event'],stage=stage,
                         evidence_from=event['evidence_from'], evidence_to=event['evidence_to'],
                         eligible_decision_times=sorted(set(eligible))))
    return rows


def audit(cases):
    results = []
    group_counts = {}
    seen = set()
    semantic_map, rules = build_map(), public_rules()
    for group,provenance,history,request_log in cases:
        seeds = tuple(provenance['seed_pair'])
        if seeds in seen:
            raise ValueError('Repeated seed pair in the observation-coverage audit')
        seen.add(seeds)
        run_results = audit_case(history)
        anchors = forecast_anchors(run_results, request_log, provenance, semantic_map, rules)
        kinds = {event['event'] for run in run_results for event in run['events']}
        counts = group_counts.setdefault(group,Counter())
        counts['episode_groups'] += 1
        counts['legal_sightings'] += len(history)
        counts['continuous_runs'] += len(run_results)
        counts['detector_eligible_runs'] += sum(row['can_fit_detector'] for row in run_results)
        for kind in ('TURN','REVERSE'):
            counts[kind+'_episode_groups'] += kind in kinds
            counts[kind+'_evidence_count'] += sum(event['event']==kind for row in run_results
                                                  for event in row['events'])
            counts[kind+'_covered_by_prior_decision'] += sum(
                row['event']==kind and bool(row['eligible_decision_times']) for row in anchors)
        results.append(dict(group=group,seed_pair=seeds,runs=run_results,decision_anchors=anchors))
    return dict(schema=SCHEMA, groups={k:dict(v) for k,v in group_counts.items()}, cases=results,
                scope='Label coverage only. A witnessed stop and heading change is not '
                      'a private patrol-state label. These are separate from frozen T1 scores.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--diagnostics',type=Path)
    parser.add_argument('--supplemental',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    cases = list(historical_cases(args.archive,args.diagnostics))
    if args.supplemental:
        cases.extend(supplemental_cases(args.supplemental))
    report = audit(cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    emit(args.output,report)
    print(json.dumps(report['groups'],indent=2))
