"""Score observable motion events and review structured behavior and Skill outputs."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
import tarfile

from llm_bundle import verify_bundle
from llm_study_cost import PRICE_REFERENCE, public_price, sum_prices
from llm_study_data import emit, read_jsonl
from llm_study_offline import LOG, MODELS, TASKS, parse, summarize
from llm_study_protocol import digest
from llm_trajectory import generate, interpolate, local_motion_envelope


EVENTS = ('MOVE', 'DWELL', 'TURN', 'REVERSE')


def detected_events(points, nodes, region):
    """Apply identical observable 10 Hz definitions to samples and predictions.

    TURN/REVERSE concern translation direction. DWELL is stationary translation,
    which does not distinguish a patrol dwell from another cause of stopping.
    """
    events = {}
    for index in range(2, len(points)-2):
        t, xy = points[index]
        before_t, before = points[index-2]
        after_t, after = points[index+2]
        a = [(x-y)/(t-before_t) for x, y in zip(xy, before)]
        b = [(x-y)/(after_t-t) for x, y in zip(after, xy)]
        va, vb = math.hypot(*a), math.hypot(*b)
        if va >= .2 and vb >= .2:
            cosine = sum(x*y for x, y in zip(a, b)) / (va * vb)
            if cosine <= math.cos(math.radians(150)):
                events.setdefault('REVERSE', t)
            elif cosine <= math.cos(math.radians(45)):
                events.setdefault('TURN', t)
        if vb >= .3:
            events.setdefault('MOVE', t)
        if max(math.dist(xy, p) for _, p in points[index-2:index+3]) <= .03:
            events.setdefault('DWELL', t)
    for (previous_t, previous), (t, xy) in zip(points, points[1:]):
        for id_, node in nodes.items():
            if math.dist(previous, node) > .3 and math.dist(xy, node) <= .3:
                events.setdefault('ENTER_NODE:'+id_, t)
        xlo, ylo = region[0]
        xhi, yhi = region[2]
        inside = lambda p: xlo <= p[0] <= xhi and ylo <= p[1] <= yhi
        if inside(previous) != inside(xy):
            events.setdefault('ENTER_REGION' if inside(xy) else 'EXIT_REGION', t)
    return events


def observation_windows(future, start, horizon):
    """At most one five-frame window per lead bin, chosen by timestamp alone."""
    leads = [(row['time']-start, row['xy']) for row in future
             if row['time']-start <= horizon + 1e-6 and row['uncertainty_m'] <= .005]
    windows = {}
    for i in range(len(leads)-4):
        points = leads[i:i+5]
        if any(abs(right[0]-left[0]-.1) > .005 for left, right in zip(points, points[1:])):
            continue
        middle = points[2][0]
        lead = '0-1s' if middle <= 1. else '1-3s' if middle <= 3. else 'over-3s'
        windows.setdefault(lead, points)
    return windows


def score_events(traces, windows, nodes, region):
    rows = []
    for lead, points in windows.items():
        actual = detected_events(points, nodes, region)
        supported = [tr for tr in traces if tr['covered_until']+1e-6 >= points[-1][0]]
        predicted = [(tr['probability'], detected_events([
            (t, interpolate(tr['states'], t)['xy']) for t, xy in points], nodes, region))
            for tr in supported]
        unknown = max(0., 1.-sum(weight for weight, _ in predicted))
        events = [*EVENTS, 'ENTER_REGION', 'EXIT_REGION', *('ENTER_NODE:'+n for n in nodes)]
        for event in events:
            truth = event in actual
            hits = [(weight, found[event]) for weight, found in predicted if event in found]
            lo = min(1., sum(w for w, _ in hits))
            hi = min(1., lo + unknown)
            timing = (sum(w*abs(t-actual[event]) for w, t in hits)/lo
                      if truth and lo > 0. else None)
            if unknown < 1e-8:
                hi = lo
            rows.append(dict(lead=lead, event=event, truth=truth, probability_lower=lo,
                             probability_upper=hi, unknown_mass=unknown,
                             timing_absolute_error_seconds=timing,
                             brier_interval=sorted(((lo-int(truth))**2, (hi-int(truth))**2)),
                             log_loss=(-math.log(max(1e-15, lo if truth else 1.-lo))
                                       if lo == hi else None)))
    return rows


def aggregate_events(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['model_id'], row['lead'], row['event'].split(':')[0]].append(row)
    reports = []
    for (model, lead, event), group in sorted(groups.items()):
        tp = sum(r['truth'] and r['probability_lower'] >= .5 for r in group)
        fp = sum(not r['truth'] and r['probability_lower'] >= .5 for r in group)
        fn = sum(r['truth'] and r['probability_upper'] < .5 for r in group)
        tn = sum(not r['truth'] and r['probability_upper'] < .5 for r in group)
        complete = [r for r in group if r['probability_lower'] == r['probability_upper']]
        timing = [r['timing_absolute_error_seconds'] for r in group
                  if r['timing_absolute_error_seconds'] is not None]
        reports.append(dict(model_id=model, lead=lead, event=event, support=len(group),
                            snapshots=len({r['sample_id'] for r in group}),
                            episode_groups=len({r['episode_id'] for r in group}),
                            positives=sum(r['truth'] for r in group),
                            predictive_coverage=len(complete), unknown_decisions=len(group)-tp-fp-fn-tn,
                            tp=tp, fp=fp, fn=fn, tn=tn,
                            precision_on_decided=tp/(tp+fp) if tp+fp else None,
                            recall_on_decided=tp/(tp+fn) if tp+fn else None,
                            recall_with_unknown_as_missed=(tp/sum(r['truth'] for r in group)
                                                           if any(r['truth'] for r in group) else None),
                            brier_score=statistics.fmean(r['brier_interval'][0] for r in complete)
                                if complete else None,
                            brier_interval=[statistics.fmean(r['brier_interval'][i] for r in group)
                                            for i in (0,1)],
                            log_loss=statistics.fmean(r['log_loss'] for r in complete)
                                if complete else None,
                            timing_support=len(timing),
                            mean_event_time_error=statistics.fmean(timing) if timing else None))
    return reports


def paired_motion(rows):
    scores = defaultdict(dict)
    for row in rows:
        key = (row['episode_id'], row['sample_id'], row['lead'], row['event'])
        scores[key][row['model_id']] = row['brier_interval']
    groups = defaultdict(lambda: defaultdict(list))
    for (episode, _, lead, event), values in scores.items():
        for a, b in combinations(sorted(values), 2):
            lo_a, hi_a = values[a]
            lo_b, hi_b = values[b]
            groups[(a,b,lead,event.split(':')[0])][episode].append((lo_b-hi_a, hi_b-lo_a))
    output = []
    for (a,b,lead,event), episodes in sorted(groups.items()):
        lower, upper = (statistics.fmean(statistics.fmean(v[i] for v in rows_in_episode)
                        for rows_in_episode in episodes.values()) for i in (0,1))
        output.append(dict(left=a, right=b, lead=lead, event=event, episode_groups=len(episodes),
                           event_windows=sum(len(v) for v in episodes.values()),
                           seed_balanced_brier_difference_interval=[lower, upper]))
    return output


def paired_outputs(matrix, sample_ids):
    rows = []
    for task in TASKS:
        for a, b in combinations(MODELS, 2):
            statuses = [(matrix.get((sample, a, task)), matrix.get((sample, b, task)))
                        for sample in sample_ids]
            paired_eligible = [(x, y) for x, y in statuses if x and y and
                               'ineligible' not in (x, y)]
            rows.append(dict(task=task, left=a, right=b, paired_eligible=len(paired_eligible),
                             both_usable=sum(x == y == 'ok' for x, y in paired_eligible),
                             usable_only_left=sum(x == 'ok' and y != 'ok' for x, y in paired_eligible),
                             usable_only_right=sum(y == 'ok' and x != 'ok' for x, y in paired_eligible),
                             unknown_left=sum(x == 'unknown' for x, y in paired_eligible),
                             unknown_right=sum(y == 'unknown' for x, y in paired_eligible)))
    return rows


def analyze(archive_path, observations, output):
    verify_bundle(archive_path)
    with tarfile.open(archive_path, 'r:gz') as archive:
        manifest = json.load(archive.extractfile('llm_run/run_manifest.json'))
        logs = read_jsonl(archive.extractfile('llm_run/'+LOG).read().decode())
    dataset = json.loads(observations.read_text())
    truth = json.loads((observations.parent/'evaluation.json').read_text())
    if (digest(dataset) != manifest['dataset']['dataset_sha256'] or
            digest(truth) != manifest['dataset']['evaluation_sha256'] or
            truth['dataset_sha256'] != digest(dataset)):
        raise ValueError('Recorded and local dataset versions differ')
    samples = {row['sample_id']: row for row in dataset['samples']}
    labels = {row['sample_id']: row for row in truth['samples']}
    predictions, behavior, actions, replay = [], [], [], []
    model_matrix = {}
    request_matrix = {}
    numeric_scored = set()
    for row in logs:
        sample_id, model, task = (row[k] for k in ('sample_id', 'model_id', 'task'))
        key = (sample_id, model, task)
        if key in model_matrix:
            raise ValueError('Duplicate matrix result')
        model_matrix[key] = row['status']
        sample, label = samples[sample_id], labels[sample_id]
        message = row['request']
        if request_matrix.setdefault((sample_id, task), message) != message:
            raise ValueError('Models did not receive identical paired task inputs')
        if message is not None and row['request_sha256'] != digest(message):
            raise ValueError('Request hash mismatch')
        raw = row.get('raw_response')
        parsed = None
        if raw is None:
            replay.append(dict(sample_id=sample_id, model_id=model, task=task, replayed=False,
                               status=row['status']))
        else:
            try:
                parsed, quality = parse(task, message, raw, dataset['map'])
                matches = (parsed == row.get('parsed') and quality == row.get('validation'))
            except (ValueError, TypeError, KeyError):
                matches = row['status'] == 'schema_error'
            replay.append(dict(sample_id=sample_id, model_id=model, task=task, replayed=True,
                               status=row['status'], matches=matches))
            if not matches:
                parsed = None
        if task == 'T1' and message:
            decision = json.loads(message[-1]['content'])['decision']
            req, track = sample['request'], sample['track']
            horizon = decision['horizon_seconds']
            region = dataset['map']['regions'][track['region']]
            nodes = {n: dataset['map']['nodes'][n]['xy'] for n in region['node_ids']}
            windows = observation_windows(label['future'], req['time'], horizon)
            traces = []
            trajectory_error = None
            if windows and parsed and any(h['state'] != 'UNKNOWN' and h['probability'] > 0
                               for h in parsed['hypotheses']):
                try:
                    traces = generate(dataset['map'], parsed, track, req['time'], horizon,
                                      dataset['rules'], mode='observed_anchor_v2')
                except ValueError as error:
                    trajectory_error = str(error)
            for scored in score_events(traces, windows, nodes, region['polygon']):
                predictions.append(dict(scored, sample_id=sample_id, model_id=model,
                                        episode_id=req['episode_id'],
                                        response_status=row['status'], trajectory_error=trajectory_error))
            if sample_id not in numeric_scored and windows:
                numeric_scored.add(sample_id)
                numerical, _ = local_motion_envelope(dataset['map'], track, req['time'],
                                                     horizon, dataset['rules'])
                for scored in score_events(numerical, windows, nodes, region['polygon']):
                    predictions.append(dict(scored, sample_id=sample_id, model_id='numeric_envelope',
                                            episode_id=req['episode_id']))
        elif task == 'T2' and parsed is not None:
            behavior.append(dict(sample_id=sample_id, model_id=model, parsed=parsed,
                                 information=sample['information'],
                                 label_status='manual_evidence_review_required'))
        elif task == 'T3' and parsed is not None:
            actions.append(dict(sample_id=sample_id, model_id=model, decision=parsed,
                                information=sample['information'],
                                numeric_recorded=label['numeric_decision']))
    cost_matrix = defaultdict(list)
    for row in logs:
        cost_matrix[row['model_id'], row['task']].append(
            public_price(row.get('raw_response'), row['wire_parameters']))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(mode=manifest['mode'], dataset=manifest['dataset'],
                  observed_matrix_rows=len(logs), expected_matrix_rows=manifest['planned'],
                  replayed=sum(r['replayed'] for r in replay),
                  replay_mismatches=[r for r in replay if r.get('matches') is False],
                  model_task_summary=summarize(logs), price_reference=PRICE_REFERENCE,
                  cost=[dict(model_id=m, task=t, **sum_prices(prices))
                        for (m,t), prices in sorted(cost_matrix.items())],
                  paired_outputs=paired_outputs(model_matrix, manifest['sample_ids']),
                  motion_metrics=aggregate_events(predictions),
                  paired_motion=paired_motion(predictions),
                  motion_window_count=len({(r['sample_id'], r['lead']) for r in predictions}),
                  available_behavior_reviews=len(behavior), available_skill_proposals=len(actions),
                  motion_scope='Five contiguous legal sightings per evaluated lead window, '
                               'same position-based event definitions on truth and trajectories. '
                               'No internal patrol-phase or torso-yaw labels.')
    for name, value in (('report', report), ('event_scores', predictions), ('replay', replay),
                        ('behavior_review', behavior), ('skill_proposals', actions)):
        emit(output/f'{name}.json', value)
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ('motion_metrics', 'model_task_summary', 'paired_outputs', 'paired_motion',
                       'cost', 'price_reference')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.archive, args.dataset, args.output)
