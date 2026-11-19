"""Run or replay the fixed-observation model matrix and package every attempt."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from llm_bundle import create_bundle, verify_bundle
from llm_decision_service import code_version
from llm_map_review import check as check_map
from llm_motion_client import (APIReasoner, PROMPT_VERSION, adapter_sources,
                               deepseek_client, qwen_client)
from llm_semantic import ModelOutputError, build_map, normalize_validated_motion, public_rules
from llm_study_data import emit, motion_messages
from llm_study_architecture import decision_packet, screen
from llm_study_protocol import (BEHAVIOR_PROMPT, BEHAVIOR_PROMPT_VERSION, BEHAVIOR_VERSION, DIRECT_PROMPT,
                                DIRECT_PROMPT_VERSION, DIRECT_VERSION, digest, messages, parse_behavior,
                                parse_direct, shared_packet, strict_json)
from online_env import ACTION_NAMES

HERE = Path(__file__).resolve().parent
MODELS = ('M1', 'M2', 'M3', 'M4')
TASKS = ('T1', 'T2', 'T3')
LOG = 'study_attempts.jsonl'


def model_configs():
    def load(provider):
        return json.loads((HERE/'config'/f'llm_decision_{provider}.json').read_text())

    deepseek, qwen = load('deepseek'), load('qwen')
    common = dict(max_tokens=8192, timeout_seconds=90, wall_timeout_seconds=90)
    configs = {
        'M1': dict(deepseek, **common),
        'M2': dict(deepseek, **common, thinking={'type': 'enabled'}, reasoning_effort='high'),
        'M3': dict(qwen, **common),
        'M4': dict(qwen, **common, enable_thinking=True, thinking_budget=4096),
    }
    configs['M4'].pop('response_format')
    return configs


def wire_parameters(config):
    if config['provider'] == 'deepseek':
        request = deepseek_client.payload(config, [])
        request.pop('messages')
    else:
        request = {key: config[key] for key in ('model', 'enable_thinking', 'max_tokens',
                   'temperature', 'thinking_budget', 'response_format') if key in config}
        request.update(stream=True, stream_options={'include_usage': True})
    return dict(request, base_url=config['base_url'], timeout_seconds=config['timeout_seconds'],
                wall_timeout_seconds=config['wall_timeout_seconds'])


def exercise(sample, dataset, task):
    if task == 'T1':
        return motion_messages(sample, dataset['map'], dataset['rules']), PROMPT_VERSION
    packet = shared_packet(sample['request'], sample['history'], dataset['map'], dataset['rules'],
                           sample['track'])
    if task == 'T2':
        track = sample['track']
        packet['decision']['observations'] = track['observations'] if track else []
        return messages(BEHAVIOR_PROMPT, packet), BEHAVIOR_PROMPT_VERSION
    packet = decision_packet(sample['request'], sample['history'], dataset['map'], dataset['rules'],
                             sample['track'], dict(max_peeks_per_stage=2, max_observation_seconds=12.))
    return messages(DIRECT_PROMPT, packet), DIRECT_PROMPT_VERSION


def parse(task, exchange, inferred, semantic_map):
    packet = json.loads(exchange[-1]['content'])
    decision = packet['decision']
    if task == 'T1':
        return normalize_validated_motion(strict_json(inferred['text']), semantic_map,
            decision['track_id'], decision['region'], decision['horizon_seconds'],
            mode='unknown_mass')
    if task == 'T2':
        return parse_behavior(inferred['text'], decision['observations']), None
    if 'limits' in decision:
        parsed = parse_direct(inferred['text'], list(ACTION_NAMES[1:]))
        public = dict(public=decision['current_robot_state'], stage=decision['stage'])
        return parsed, screen(parsed, public, decision, enabled=False)
    return parse_direct(inferred['text'], decision['allowed_skills']), None


def fixture_response(task, exchange):
    from llm_decision_service import fixture_response as fixture_motion

    payload = json.loads(exchange[-1]['content'])
    if task == 'T1':
        result = fixture_motion(payload, compact=True)
    elif task == 'T2':
        from llm_study_protocol import BEHAVIOR_KINDS
        result = dict(schema=BEHAVIOR_VERSION, facts=[], assessments=[
            dict(kind=kind, status='UNKNOWN', evidence_times=[],
                 inference='Fixture response', uncertainty='Fixture response')
            for kind in sorted(BEHAVIOR_KINDS)])
    else:
        result = dict(schema=DIRECT_VERSION, skill=payload['decision']['allowed_skills'][0],
                      reason='Fixture response', uncertainty='Fixture response')
    return dict(text=json.dumps(result), usage=None, model='fixture-only',
                response_source='fixture', wall_seconds=0., raw_metadata={})


def distributions(values):
    values = sorted(values)
    if not values:
        return None
    return dict(n=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                p90=values[max(0, (9*len(values)+9)//10 - 1)], max=values[-1])


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['model_id'], row['task']), []).append(row)
    summary = []
    for (model, task), group in groups.items():
        received = [row for row in group if row.get('raw_response') is not None]
        usages = [row['raw_response'].get('usage') or {} for row in received]
        token_fields = ('prompt_tokens', 'completion_tokens', 'total_tokens')
        tokens = {key: dict(sum_reported=sum(u[key] for u in usages if type(u.get(key)) is int),
                            reported_calls=sum(type(u.get(key)) is int for u in usages))
                  for key in token_fields}
        reasoning = [(u.get('completion_tokens_details') or {}).get('reasoning_tokens')
                     for u in usages]
        tokens['reasoning_tokens'] = dict(sum_reported=sum(v for v in reasoning if type(v) is int),
                                         reported_calls=sum(type(v) is int for v in reasoning))
        summary.append(dict(model_id=model, task=task, trials=len(group),
                            status=dict(Counter(row['status'] for row in group)),
                            usage=tokens, api=distributions([
                                row['api_wall_seconds'] for row in group
                                if row.get('api_wall_seconds') is not None]),
                            local=distributions([
                                row['local_wall_seconds'] for row in group
                                if row.get('local_wall_seconds') is not None])))
    return summary


def attempt(sample, task, model_id, config, prepared, reasoner, fixture, dataset):
    exchange, version = prepared
    request = sample['request']
    row = dict(sample_id=sample['sample_id'], task=task, model_id=model_id,
               episode_id=request['episode_id'], request_id=request['request_id'],
               game_time=request['time'], map_id=dataset['map']['map_id'],
               rules_version=dataset['rules']['version'],
               started_utc=datetime.now(timezone.utc).isoformat(),
               model_requested=config['model'],
               thinking=(config.get('thinking', {}).get('type') == 'enabled' or
                         config.get('enable_thinking', False)), prompt_version=version,
               wire_parameters=wire_parameters(config), request=exchange,
               request_sha256=digest(exchange) if exchange else None)
    if exchange is None:
        return dict(row, status='ineligible', error='No recent associated two-sample track')
    started = time.perf_counter()
    api_finished = None
    try:
        result = (fixture_response(task, exchange) if fixture else
                  reasoner.infer(exchange))
        api_finished = time.perf_counter()
        row['raw_response'] = result
        row['api_wall_seconds'] = result.get('wall_seconds')
        row['api_client_wall_seconds'] = api_finished - started
    except (ValueError, OSError, qwen_client.APIError) as error:
        elapsed = time.perf_counter()-started
        return dict(row, status='api_error', api_client_wall_seconds=elapsed,
                    api_wall_seconds=elapsed if isinstance(error, qwen_client.APIError) else None,
                    error_type=type(error).__name__, error=str(error))
    return dict(row, status='received', received_at=api_finished)


def run(dataset_path, output, fixture=False, smoke=False):
    dataset = json.loads(dataset_path.read_text())
    manifest = json.loads((dataset_path.parent/'dataset_manifest.json').read_text())
    if digest(dataset) != manifest['dataset_sha256']:
        raise ValueError('Frozen dataset hash mismatch')
    if dataset['map'] != build_map() or dataset['rules'] != public_rules():
        raise ValueError('Current map or rules differ from the frozen observations')
    for sample in dataset['samples']:
        from llm_decision_core import _check_public_request
        _check_public_request(sample['request'])
    samples = dataset['samples']
    fresh = next(s for s in samples if s['information'] == 'sufficient'
                 and s['phase'] == 'before_gate')
    missing = next(s for s in samples if s['information'] == 'insufficient'
                   and s['request']['episode_id'] != fresh['request']['episode_id'])
    development_episodes = {s['request']['episode_id'] for s in (fresh, missing)}
    selected = ([fresh, missing] if smoke else [s for s in samples
                if s['request']['episode_id'] not in development_episodes])
    prepared = {(s['sample_id'], task): exercise(s, dataset, task)
                for s in selected for task in TASKS}
    configs = model_configs()
    reasoners = {model: APIReasoner(config) for model, config in configs.items()}
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = output.parent / (output.name + '.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    report = dict(schema='llm-study-offline-run-v1', mode='fixture' if fixture else
                  'development_api_smoke' if smoke else 'offline_historical',
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  code_version=code_version(), adapter_sources=adapter_sources(),
                  study_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(HERE.glob('llm_study*.py'))},
                  dataset=manifest, map_review_status=check_map()['review']['status'],
                  development_episodes=sorted(development_episodes),
                  sample_ids=[s['sample_id'] for s in selected],
                  configs={name: wire_parameters(v) for name, v in configs.items()},
                  attempted=0, planned=len(selected)*len(TASKS)*len(MODELS),
                  maximum_live_api_calls=0 if fixture else len(MODELS)*sum(
                      exchange is not None for exchange, version in prepared.values()),
                  api_elapsed_definition='Client HTTPS elapsed from provider adapter; '
                                         'server-only time unavailable.',
                  error=None)
    emit(output/'run_manifest.json', report)
    rows = []
    try:
        with (output/LOG).open('x') as log:
            for index, sample in enumerate(selected):
                order = MODELS[index % len(MODELS):] + MODELS[:index % len(MODELS)]
                for task in TASKS:
                    request = prepared[sample['sample_id'], task]
                    for model in order:
                        row = attempt(sample, task, model, configs[model], request,
                                      reasoners[model], fixture, dataset)
                        if row['status'] == 'received':
                            started = row.pop('received_at')
                            try:
                                parsed, quality = parse(task, row['request'], row['raw_response'],
                                                        dataset['map'])
                                row.update(parsed=parsed, validation=quality, status='ok')
                                if (task == 'T3' and parsed['skill'] == 'UNKNOWN' or
                                        task == 'T2' and all(a['status'] == 'UNKNOWN'
                                                             for a in parsed['assessments']) or
                                        task == 'T1' and all(h['state'] == 'UNKNOWN'
                                                             for h in parsed['hypotheses'])):
                                    row['status'] = 'unknown'
                                elif task == 'T3' and quality and quality['rejection']:
                                    row['status'] = 'constraint_violation'
                            except (ModelOutputError, ValueError, KeyError, TypeError) as error:
                                row.update(status='schema_error', error_type=type(error).__name__,
                                           error=str(error))
                            row['local_wall_seconds'] = time.perf_counter() - started
                        log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        log.flush()
                        rows.append(row)
                print(f"Completed {index+1}/{len(selected)} observation samples", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = f'{type(error).__name__}: {error}'
    report.update(attempted=len(rows), finished_utc=datetime.now(timezone.utc).isoformat(),
                  status='failed' if report['error'] else 'completed',
                  results=summarize(rows))
    emit(output/'run_manifest.json', report)
    secrets = [os.environ.get(config['api_key_env']) for config in configs.values()]
    create_bundle(output, archive, secrets, additional_files=(LOG,))
    print(json.dumps(dict(status=report['status'], attempted=len(rows),
                          maximum_live_api_calls=report['maximum_live_api_calls'],
                          upload_this=str(archive), error=report['error'])), flush=True)
    return 1 if report['error'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Two-sample API integration check')
    parser.add_argument('--fixture-for-tests', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_bundle(args.verify), indent=2))
        return 0
    if not args.dataset or not args.output:
        parser.error('--dataset and --output are required')
    return run(args.dataset, args.output, args.fixture_for_tests, args.smoke)


if __name__ == '__main__':
    raise SystemExit(main())
