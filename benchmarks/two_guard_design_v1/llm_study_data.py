"""Freeze paired offline samples from historical public decision logs."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile

from llm_bundle import verify_bundle
from llm_compare_ubuntu import verify_comparison
from llm_decision_core import EpisodeContext, _check_public_request
from llm_motion_client import messages_for
from llm_robot_rollout import prediction_horizon
from llm_semantic import SCHEMA_VERSION, build_map, public_rules
from llm_study_protocol import digest


def read_jsonl(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def emit(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def archived_logs(source):
    verify_comparison(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with tarfile.open(source, 'r:gz') as archive, tempfile.TemporaryDirectory() as directory:
        comparison = json.load(archive.extractfile('llm_comparison/comparison.json'))
        for index, seeds in enumerate(comparison['seed_pairs']):
            name = f'llm_comparison/{index:03d}-numeric.tar.gz'
            data = archive.extractfile(name).read()
            path = Path(directory) / 'episode.tar.gz'
            path.write_bytes(data)
            verify_bundle(path)
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as inner:
                log = inner.extractfile('llm_run/decision_events.jsonl').read()
            yield read_jsonl(log), dict(archive_sha256=source_hash, member=name,
                                        log_sha256=hashlib.sha256(log).hexdigest(),
                                        seed_pair=seeds)


def diagnostic_logs(root):
    for index in range(16):
        path = root / f'fresh-1-1-{index:03d}' / 'decision_events.jsonl'
        log = path.read_bytes()
        yield read_jsonl(log), dict(member=str(path.relative_to(root)),
                                    log_sha256=hashlib.sha256(log).hexdigest(),
                                    seed_pair=[1010100+17*index, 1020100+19*index])


def episode_candidates(events, provenance, semantic_map, rules, group):
    context = EpisodeContext(group, semantic_map['map_id'], rules['version'],
                             velocity_mode='recent_acceleration_v2')
    seen = {}
    candidates = []
    for event in events:
        if event['kind'] == 'finish':
            for row in event['payload']['observations']:
                seen[(row['stage'], row['time'])] = row
            continue
        if event['kind'] != 'decision':
            continue
        request = event['request']
        _check_public_request(request)
        context.add_samples(request['observations'])
        for row in request['observations']:
            seen[(row['stage'], row['time'])] = row
        time = request['time']
        stage = request['stage']
        track = context.track_for(stage)
        track = dict(track, observations=list(track['observations'])) if track else None
        age = time - track['observations'][-1]['time'] if track else None
        if not track or len(track['observations']) < 2 or age > 4.:
            information = 'insufficient'
        elif age > 1.5 or track['association_uncertainty'] or (
                sum(v*v for v in track['estimated_velocity']) < .15**2):
            information = 'ambiguous'
        else:
            information = 'sufficient'
        phase = ('boundary' if request['public']['action_mask'][7] else
                 'initial' if request['public']['peek_count'] == 0 else 'before_gate')
        history = [row for _, row in sorted(seen.items())]
        candidates.append(dict(request=request, history=history, track=track,
                               phase=phase, information=information, group=group,
                               provenance=dict(provenance, source_request_id=request['request_id']),
                               numeric_outcome=event['response']))
    for sample in candidates:
        request = sample['request']
        sample['future'] = [row for (stage, time), row in sorted(seen.items())
                            if stage == request['stage'] and time > request['time']+1e-5]
    return candidates


def select(candidates, count):
    """Round-robin strata and seed groups without consulting future outcomes."""
    available = list(candidates)
    strata, groups = Counter(), Counter()
    chosen = []
    while available and len(chosen) < count:
        def priority(sample):
            request = sample['request']
            stratum = (request['stage'], sample['phase'], sample['information'])
            return (groups[sample['group']] >= 2, strata[stratum],
                    groups[sample['group']], digest([sample['group'], request['time']]))
        selected = min(range(len(available)), key=lambda index: priority(available[index]))
        sample = available.pop(selected)
        request = sample['request']
        strata[(request['stage'], sample['phase'], sample['information'])] += 1
        groups[sample['group']] += 1
        chosen.append(sample)
    return chosen


def motion_messages(sample, semantic_map, rules):
    request, track = sample['request'], sample['track']
    if not track or len(track['observations']) < 2:
        return None
    last = track['observations'][-1]
    age = request['time'] - last['time']
    if age > 4.:
        return None
    horizon = prediction_horizon(request['public'], request['stage'])
    recent = track['observations'][-30:]
    reasoning = dict(schema=SCHEMA_VERSION, request_id=request['request_id'],
                     map_id=semantic_map['map_id'], track_id=track['id'],
                     region=track['region'], game_time=request['time'],
                     horizon_seconds=horizon, latest_measurement_at=last['time'],
                     measurement_age_seconds=age, path_coverage_seconds=horizon+age,
                     current_robot_state=request['public'], skill_state=request['skill_state'],
                     recent_observations=recent,
                     older_observation_count=len(track['observations'])-len(recent),
                     estimated_guard_velocity=track['estimated_velocity'],
                     observable_guard_facing=None,
                     association_uncertainty=track['association_uncertainty'],
                     previous_hypotheses=[], prediction_comparisons=[])
    return messages_for(reasoning, semantic_map, rules)


def freeze(source, history_dir, output, count):
    semantic_map, rules = build_map(), public_rules()
    candidates = []
    logs = list(archived_logs(source))
    if history_dir:
        logs.extend(diagnostic_logs(history_dir))
    for index, (events, provenance) in enumerate(logs):
        candidates.extend(episode_candidates(events, provenance, semantic_map, rules,
                                             f'historical-{index:03d}'))
    output.mkdir(parents=True, exist_ok=False)
    inputs, evaluation = [], []
    for index, sample in enumerate(select(candidates, count)):
        sample_id = f'observation-{index+1:03d}'
        request = dict(sample['request'], request_id=sample_id, episode_id=sample['group'],
                       skill_state=dict(sample['request']['skill_state']))
        prior = request['skill_state'].get('plan_request_id')
        if prior is not None:
            request['skill_state']['plan_request_id'] = 'prior-public-plan'
        _check_public_request(request)
        public = {key: sample[key] for key in ('history', 'track', 'phase', 'information')}
        public.update(sample_id=sample_id, request=request)
        inputs.append(public)
        response = sample['numeric_outcome']
        evaluation.append(dict(sample_id=sample_id, provenance=sample['provenance'],
                               future=sample['future'],
                               numeric_decision={key: response[key] for key in
                                  ('status', 'action', 'plan', 'gate', 'error_type', 'timing')
                                  if key in response}))
    dataset = dict(schema='llm-study-observations-v1', split='offline_historical',
                   map=semantic_map, rules=rules, samples=inputs)
    labels = dict(schema='llm-study-evaluation-v1', dataset_sha256=digest(dataset),
                  samples=evaluation)
    emit(output/'observations.json', dataset)
    emit(output/'evaluation.json', labels)
    manifest = dict(dataset_sha256=digest(dataset), evaluation_sha256=digest(labels),
                    source_archive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    episodes=len({sample['request']['episode_id'] for sample in inputs}),
                    source_episodes=len(logs), source_candidates=len(candidates),
                    samples=len(inputs),
                    strata=dict(Counter(f"{s['request']['stage']}/{s['phase']}/{s['information']}"
                                        for s in inputs)),
                    map_id=semantic_map['map_id'], rules_version=rules['version'],
                    map_review_status=json.loads((Path(__file__).parent/'llm_map_manifest.json')
                                                 .read_text())['review']['status'])
    emit(output/'dataset_manifest.json', manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--historical-diagnostics', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=40)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count must be positive')
    freeze(args.source, args.historical_diagnostics, args.output, args.count)
