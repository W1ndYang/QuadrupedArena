"""Compare matched snapshot actions and replay one Option against the formal referee."""
import argparse
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import tarfile

from llm_bundle import verify_bundle
from llm_decision_core import DecisionOrchestrator, EpisodeContext
from llm_option_policy import LLMOptionPolicy
from llm_semantic import ModelOutputError
from llm_study_architecture import decision_packet, screen
from llm_study_data import archived_logs, diagnostic_logs, emit, read_jsonl
from llm_study_offline import LOG, MODELS, HERE
from llm_study_protocol import digest, parse_direct
from online_env import ACTION_NAMES
from option_env import run_episode


class BoundaryReached(Exception):
    pass


class ApplyIntervention(Exception):
    pass


class PrefixReplay:
    def __init__(self, events, sample, action):
        self.events = [row for row in events if row['kind'] == 'decision']
        self.sample_id = sample
        self.action = ACTION_NAMES.index(action)
        self.index = 0
        self.applied = False

    def decide(self, request):
        if self.index >= len(self.events):
            raise ValueError('Historical decision prefix ended before the snapshot')
        expected = self.events[self.index]
        self.index += 1
        if request != expected['request']:
            raise ValueError('Historical prefix does not reconstruct the exact decision request')
        if request['request_id'] != self.sample_id:
            return expected['response']
        self.applied = True
        raise ApplyIntervention


class OneOptionPolicy(LLMOptionPolicy):
    def __call__(self, observation, mask):
        if self.client.applied:
            raise BoundaryReached
        try:
            return super().__call__(observation, mask)
        except ApplyIntervention:
            return self._apply_direct_skill(
                dict(status='skill', action=self.client.action), mask)


def replay_option(events, provenance, skill):
    first = next(event for event in events if event['kind'] == 'decision')
    client = PrefixReplay(events, provenance['source_request_id'], skill)
    policy = OneOptionPolicy(first['request']['episode_id'], client, 48.)
    try:
        summary, records = run_episode(*provenance['seed_pair'], policy)
    except BoundaryReached as returned:
        summary, records = None, returned.closed_options
    if not client.applied or not records:
        raise ValueError('Replay ended before the requested Option')
    last = records[-1]
    if ACTION_NAMES[last['action']] != skill:
        raise ValueError('Counterfactual Option differs from selected Skill')
    observed = [row for row in policy.samples if row['time'] > last['time']+1e-5]
    return dict(skill=skill, start=last['time'], end=last['end_time'],
                termination=last['termination_reason'],
                winner=summary['winner'] if summary else None,
                formal_reason=summary['reason'] if summary else None,
                new_legal_sightings=len(observed),
                scope='observed referee result up to the selected Option return')


def first_skill(result):
    if result.get('action') is not None:
        return ACTION_NAMES[result['action']]
    plan = result.get('plan')
    if not plan:
        return None
    if plan['wait']:
        return 'WAIT'
    return ('SMOKE_'+plan['side'].upper() if plan['smoke'] else
            plan['side'].upper()+'_GATE')


class SnapshotReasoner:
    def __init__(self, row):
        self.row = row
        self.used = False

    def infer(self, exchange):
        self.used = True
        if self.row is None or self.row.get('raw_response') is None:
            raise ModelOutputError('No matched offline motion response for this state')
        # The legacy D1 prompt at a fresh snapshot has no policy-specific
        # prediction history. Verify exact equality before accepting a replay.
        if exchange != self.row['request']:
            recorded = self.row['request']
            left, right = (json.loads(messages[-1]['content']) for messages in (exchange, recorded))
            # The frozen dataset and D1 associate the same public time values
            # with different floating-point parenthesization.
            coverage = 'path_coverage_seconds'
            a, b = (packet['decision'].pop(coverage) for packet in (left, right))
            if exchange[0] != recorded[0] or left != right or not math.isclose(
                    a, b, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Snapshot motion request differs from the recorded prompt')
        return self.row['raw_response']


def prediction_snapshot(sample, row, dataset):
    config = json.loads((HERE/'config'/'llm_decision_deepseek.json').read_text())
    reasoner = SnapshotReasoner(row)
    orchestrator = DecisionOrchestrator(reasoner, config)
    request = sample['request']
    context = EpisodeContext(request['episode_id'], dataset['map']['map_id'],
                             dataset['rules']['version'], velocity_mode=config['velocity_mode'])
    batch = {(r['stage'], r['time']) for r in request['observations']}
    context.add_samples([r for r in sample['history'] if (r['stage'],r['time']) not in batch])
    orchestrator.episodes[request['episode_id']] = context
    try:
        result = orchestrator.decide(request)
    except (ValueError, ModelOutputError) as error:
        return dict(status='error', skill=None, error_type=type(error).__name__, error=str(error))
    return dict(status=result['status'], skill=first_skill(result), plan=result.get('plan'),
                gate=result.get('gate'), timing=result.get('timing'),
                model_called_or_replayed=reasoner.used)


def direct_snapshot(row, request, decision, safety):
    raw = row.get('raw_response')
    if raw is None:
        return dict(status=row.get('status', 'unattempted'), skill=None)
    try:
        parsed = parse_direct(raw['text'], list(ACTION_NAMES[1:]))
        checked = screen(parsed, request, decision, enabled=safety)
        return dict(checked, skill=checked['executed'], parsed=parsed,
                    api_seconds=row.get('api_wall_seconds'))
    except ModelOutputError as error:
        return dict(status='schema_error', skill=None, error=str(error))


def immediate_summary(rows):
    output = {}
    for row in rows:
        for method, decision in row['methods'].items():
            stats = output.setdefault(method, dict(states=0, actions=0, evaluated_options=0,
                                                   terminal_wins=0, terminal_losses=0,
                                                   safety_rejections=0, useful_peeks=0,
                                                   gate_crossings=0, returned_to_cover=0))
            stats['states'] += 1
            skill = decision.get('skill')
            stats['actions'] += skill is not None
            stats['safety_rejections'] += decision.get('rejection') is not None
            ref = row.get('counterfactuals', {}).get(skill)
            if ref and 'error' not in ref:
                stats['evaluated_options'] += 1
                stats['terminal_wins'] += ref['winner'] == 'attacker'
                stats['terminal_losses'] += ref['winner'] == 'defender'
                stats['useful_peeks'] += (ref['termination'] == 'peek_returned'
                                         and ref['new_legal_sightings'] >= 2)
                stats['gate_crossings'] += ref['termination'] == 'gate_crossed'
                stats['returned_to_cover'] += ref['termination'] == 'cover_reached'
    return output


def paired_immediate(rows):
    if not rows:
        return []
    methods = list(rows[0]['methods'])
    result = []
    for left,right in combinations(methods,2):
        for phase in ('all','before_gate','boundary','initial'):
            paired = []
            for row in rows:
                if phase != 'all' and phase != row['phase']:
                    continue
                skill_a,skill_b = (row['methods'][method]['skill'] for method in (left,right))
                outcomes = row.get('counterfactuals',{})
                a,b = (outcomes.get(skill) for skill in (skill_a,skill_b))
                if not a or not b or 'error' in a or 'error' in b:
                    continue
                paired.append((row['episode_id'],skill_a!=skill_b,
                               a['winner']=='defender',b['winner']=='defender'))
            if paired:
                result.append(dict(left=left,right=right,phase=phase,snapshots=len(paired),
                    seed_groups=len({r[0] for r in paired}),
                    different_actions=sum(r[1] for r in paired),
                    both_fail=sum(r[2] and r[3] for r in paired),
                    left_only_fails=sum(r[2] and not r[3] for r in paired),
                    right_only_fails=sum(not r[2] and r[3] for r in paired)))
    return result


def compare(archive, data_path, output, direct_model, motion_model, source=None,
            diagnostic_root=None, counterfactual_count=0):
    verify_bundle(archive)
    with tarfile.open(archive) as package:
        manifest = json.load(package.extractfile('llm_run/run_manifest.json'))
        logs = read_jsonl(package.extractfile('llm_run/'+LOG).read().decode())
    dataset = json.loads(data_path.read_text())
    labels = json.loads((data_path.parent/'evaluation.json').read_text())
    if digest(dataset) != manifest['dataset']['dataset_sha256'] or (
            digest(labels) != manifest['dataset']['evaluation_sha256']):
        raise ValueError('Dataset hash mismatch')
    by_request = {(r['sample_id'], r['task'], r['model_id']): r for r in logs}
    evaluation = {row['sample_id']: row for row in labels['samples']}
    prefix_logs = {}
    if counterfactual_count:
        for log, provenance in archived_logs(source):
            prefix_logs[provenance['log_sha256']] = log
        if diagnostic_root:
            for log, provenance in diagnostic_logs(diagnostic_root):
                prefix_logs[provenance['log_sha256']] = log
    rows = []
    for sample in dataset['samples']:
        sample_id = sample['sample_id']
        if sample_id not in manifest['sample_ids']:
            continue
        label = evaluation[sample_id]
        numeric = label['numeric_decision']
        motion = by_request.get((sample_id, 'T1', motion_model))
        req = sample['request']
        packet = decision_packet(req, sample['history'], dataset['map'], dataset['rules'],
                                 sample['track'], dict(max_peeks_per_stage=2,
                                                       max_observation_seconds=12.))
        decisions = dict(D0=dict(numeric, skill=first_skill(numeric)),
                         D1=prediction_snapshot(sample, motion, dataset))
        for model in (MODELS if direct_model == 'all' else (direct_model,)):
            direct = by_request.get((sample_id, 'T3', model), {})
            for arm in ('D2','D3'):
                name = f'{arm}_{model}' if direct_model == 'all' else arm
                decisions[name] = direct_snapshot(direct, req, packet['decision'],
                                                  safety=arm=='D3')
        row = dict(sample_id=sample_id, episode_id=req['episode_id'], time=req['time'],
                   phase=sample['phase'], information=sample['information'], methods=decisions)
        if len(rows) < counterfactual_count:
            provenance = label['provenance']
            prefix = prefix_logs.get(provenance['log_sha256'])
            if prefix is None:
                row['counterfactual_unavailable'] = 'Historical source log unavailable'
            else:
                results = {}
                for skill in sorted({d['skill'] for d in decisions.values() if d['skill']}):
                    try:
                        results[skill] = replay_option(prefix, provenance, skill)
                    except ValueError as error:
                        results[skill] = dict(error=str(error))
                row['counterfactuals'] = results
        rows.append(row)
    report = dict(fixture_only=manifest['mode']=='fixture', source_sha256=hashlib.sha256(
        archive.read_bytes()).hexdigest(), direct_model=direct_model, motion_model=motion_model,
        method_notes=dict(D0='Actual recorded numerical decision',
                          D1='Legacy prediction/optimization with a cold prediction cache',
                          D2='Direct proposal with executor contract',
                          D3='Same proposal with hard checks and bounded fallback'),
        rows=rows, immediate_option_summary=immediate_summary(rows),
        paired_immediate=paired_immediate(rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    emit(output, report)
    print(json.dumps(dict(snapshots=len(rows), counterfactual_count=sum(
        len(r.get('counterfactuals', {})) for r in rows), output=str(output))))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--direct-model', choices=(*MODELS, 'all'), required=True)
    parser.add_argument('--motion-model', choices=MODELS, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--historical-diagnostics', type=Path)
    parser.add_argument('--counterfactual-count', type=int, default=0)
    args = parser.parse_args()
    if args.counterfactual_count and not args.source:
        parser.error('--source required for formal counterfactual replay')
    compare(args.archive, args.dataset, args.output, args.direct_model, args.motion_model,
            args.source, args.historical_diagnostics, args.counterfactual_count)
