"""Paired full-episode evaluation through the existing formal episode runner."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

from scipy.stats import binomtest

from llm_compare_ubuntu import comparison_bundle, read_result, verify_comparison
from llm_decision_service import code_version
from llm_motion_client import _key
from llm_run_ubuntu import _git_head, run as run_episode_bundle
from llm_study_data import emit, read_jsonl
from llm_study_offline import MODELS, distributions, model_configs, wire_parameters
from llm_study_protocol import digest


HERE = Path(__file__).resolve().parent
METHODS = ('D0', 'D1', 'D1_matched', 'D2', 'D3')
PILOT_PLAN = HERE/'config'/'llm_study_pilot_seeds.json'
EVENT_PLAN = HERE/'config'/'llm_study_turn_collection.json'
FIXTURE_SEEDS = [810000,820000]


def excluded_seed_streams(phase):
    banned_first = {810000, *(810100+17*i for i in range(8)),
                    *(1010100+17*i for i in range(16))}
    banned_second = {820000, *(820100+19*i for i in range(8)),
                     *(1020100+19*i for i in range(16))}
    development = json.loads(EVENT_PLAN.read_text())['seed_pairs']
    banned_first.update(pair[0] for pair in development)
    banned_second.update(pair[1] for pair in development)
    if phase == 'holdout':
        pilot = load_plan(PILOT_PLAN, 'pilot')
        banned_first.update(pair[0] for pair in pilot['seed_pairs'])
        banned_second.update(pair[1] for pair in pilot['seed_pairs'])
    return banned_first, banned_second


def load_plan(path, phase):
    plan = json.loads(path.read_text())
    if plan['phase'] != phase or plan['schema'] != 'llm-study-seeds-v1':
        raise ValueError('Seed plan phase or schema mismatch')
    pairs = plan['seed_pairs']
    if not isinstance(pairs, list) or not pairs or any(
            not isinstance(pair, list) or len(pair) != 2 or
            any(type(n) is not int or not 0 <= n < 2**32 for n in pair) for pair in pairs):
        raise ValueError('Invalid guard seeds')
    banned_first, banned_second = excluded_seed_streams(phase)
    for first, second in pairs:
        if first in banned_first or second in banned_second:
            raise ValueError('Seed repeats a previous or planned episode')
        banned_first.add(first)
        banned_second.add(second)
    return plan


def config_matrix(direct_model):
    def from_file(name):
        return json.loads((HERE/'config'/f'llm_decision_{name}.json').read_text())
    original = from_file('deepseek')
    representative = original if direct_model == 'M1' else model_configs()[direct_model]
    configs = {
        'D0': from_file('numeric'),
        'D1': original,
        'D1_matched': dict(original, study_architecture='D1_matched'),
    }
    if direct_model != 'M1':
        configs['D1_selected'] = dict(representative, study_architecture='D1_matched')
    configs.update(D2=dict(representative, study_architecture='D2', direct_call_cap_per_stage=12),
                   D3=dict(representative, study_architecture='D3', direct_call_cap_per_stage=12))
    return configs


def enrich(row, directory):
    path = directory/'decision_events.jsonl'
    events = read_jsonl(path.read_text()) if path.exists() else []
    decisions = [e for e in events if e['kind'] == 'decision']
    error = next((e for e in reversed(decisions) if e['response']['status'] == 'error'), None)
    if error and error['response'].get('error_type') in ('APIError', 'FileNotFoundError'):
        row.update(status='api_failure', failure_category='api', summary=None)
    elif error and row['status'] == 'policy_failure':
        reason = error['response'].get('error', '')
        if reason.startswith('Direct model decision budget'):
            row['failure_category'] = 'model_budget'
        if reason.startswith('Direct proposal rejected:'):
            row['failure_category'] = 'rejected_skill'
    summary = row['summary'] or {}
    manifest = json.loads((directory/'run_manifest.json').read_text())
    path = directory/f"{manifest['episode_id']}.json"
    episode = json.loads(path.read_text()) if path.exists() else {}
    wall_seconds = ((datetime.fromisoformat(episode['finished_utc']) -
                     datetime.fromisoformat(episode['started_utc'])).total_seconds()
                    if episode.get('finished_utc') else None)
    options = episode.get('options', [])
    if not options and row['status'] != 'completed':
        options = episode.get('options_before_failure', [])
    action_counts = Counter(option['action'] for option in options)
    exchanges = [e for e in decisions if e.get('llm_request') is not None]
    responses = [e['response'].get('reasoning', {}).get('raw_response') or
                 e.get('llm_response') for e in exchanges]
    usages = [v.get('usage') or {} for v in responses if v]
    checks = [e['response'].get('reasoning', {}).get('safety_check') or
              (e.get('llm_response') or {}).get('safety_check') for e in decisions]
    row.update(formal_terminal=row['status'] == 'completed', official_reason=(
                   summary.get('reason') if row['status'] == 'completed' else None),
               decisions=len(decisions), action_counts=dict(action_counts),
               policy_failure_replay_matches=manifest.get('policy_failure_replay_matches'),
               episode_wall_seconds=wall_seconds,
               terminal_option=options[-1] if options else None,
               option_termination_counts=dict(Counter(v['termination'] for v in options)),
               referee_telemetry=summary.get('referee_telemetry', episode.get('referee_telemetry')),
               observation_times=[v['start'] for v in options if v['action'].startswith('PEEK_')],
               smoke_used=summary.get('smoke_used'),
               plan_changes=sum(a.get('plan') is not None and b.get('plan') is not None and
                   a['plan'] != b['plan'] for a, b in zip(episode.get('decisions', []),
                                                       episode.get('decisions', [])[1:])),
               safety_checks=[v for v in checks if v],
               total_tokens_reported=sum(u['total_tokens'] for u in usages
                                         if type(u.get('total_tokens')) is int),
               calls_with_total_tokens=sum(type(u.get('total_tokens')) is int for u in usages),
               full_service_seconds=[e['service_wall_seconds'] for e in decisions],
               local_compute_seconds=[max(0., e['service_wall_seconds'] -
                     float((e['response'].get('reasoning', {}).get('raw_response') or
                            e.get('llm_response') or {}).get('wall_seconds') or 0.))
                     for e in decisions if e['response'].get('error_type') != 'APIError'])
    return row


def wilson(wins, n):
    if not n:
        return None
    z = 1.95996398454
    p = wins/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., center-radius), min(1., center+radius)]


def success(row):
    return row['status'] == 'completed' and row['summary']['winner'] == 'attacker'


def paired(rows, methods, count, include_api_failures=True):
    lookup = {(r['index'], r['method']): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate method result for a paired seed index')
    seeds = {}
    for row in rows:
        index, method = row['index'], row['method']
        pair = (row['first_seed'],row['second_seed'])
        if (method not in methods or not 0<=index<count or seeds.setdefault(index,pair)!=pair):
            raise ValueError('Result differs from the paired method or seed set')
    if len(set(seeds.values())) != len(seeds):
        raise ValueError('Repeated seed pair across comparison indices')
    included = ('completed','policy_failure','api_failure') if include_api_failures else (
                'completed','policy_failure')
    report = {}
    for a, b in combinations(methods, 2):
        values = []
        for index in range(count):
            left, right = lookup.get((index,a)), lookup.get((index,b))
            if left and right and all(r['status'] in included for r in (left,right)):
                values.append((success(left),success(right)))
        if not values:
            continue
        n10 = sum(x and not y for x, y in values)
        n01 = sum(y and not x for x, y in values)
        discordant = n10+n01
        p = binomtest(n01, discordant).pvalue if discordant else 1.
        delta = [int(y)-int(x) for x,y in values]
        gain = binomtest(n01, len(values)).proportion_ci(confidence_level=.975)
        loss = binomtest(n10, len(values)).proportion_ci(confidence_level=.975)
        report[f'{b}_vs_{a}'] = dict(n=len(values), unavailable=count-len(values),
            both_win=sum(x and y for x,y in values), both_fail=sum(not x and not y for x,y in values),
            reference_only=n10, comparison_only=n01, exact_two_sided_p=p,
            paired_difference=statistics.fmean(delta),
            conservative_paired_difference_95=[gain.low-loss.high, gain.high-loss.low])
    return report


def primary_comparisons(methods):
    matched = 'D1_selected' if 'D1_selected' in methods else 'D1_matched'
    return ('D2_vs_D0','D3_vs_D2',f'D2_vs_{matched}')


def holm_primary(report, names):
    ordered = sorted((report[name]['exact_two_sided_p'], name)
                     for name in names if name in report)
    adjustment = {name: None for name in names}
    bound = 0.
    for index,(p,name) in enumerate(ordered):
        bound = max(bound,min(1.,p*(len(names)-index)))
        adjustment[name] = bound
    return adjustment


def aggregate(rows, methods, count):
    result = {}
    for method in methods:
        group = [row for row in rows if row['method'] == method]
        wins = sum(success(r) for r in group)
        n = len(group)
        outcome = Counter(r['official_reason'] or r['failure_category'] or 'run_error'
                          for r in group)
        result[method] = dict(wins=wins, attempted=n, planned=count,
            status=dict(Counter(r['status'] for r in group)), outcome=dict(outcome),
            success_rate=wins/n if n else None, wilson_95=wilson(wins,n),
            mean_winning_seconds=statistics.fmean(r['summary']['seconds'] for r in group
                                                  if success(r)) if wins else None,
            game_seconds=distributions([r['summary']['seconds'] for r in group
                                        if r['formal_terminal']]),
            wall_seconds=distributions([r['episode_wall_seconds'] for r in group
                                        if r['episode_wall_seconds'] is not None]),
            model_calls=sum(r['llm_inference_attempts'] for r in group),
            total_tokens_reported=sum(r['total_tokens_reported'] for r in group),
            safety_rejections=sum(c['rejection'] is not None for r in group for c in r['safety_checks']),
            api_seconds=distributions([v for r in group for v in r['model_seconds'] if v is not None]),
            service_seconds=distributions([v for r in group for v in r['full_service_seconds']]),
            local_seconds=distributions([v for r in group for v in r['local_compute_seconds']]))
    comparisons = paired(rows, methods, count)
    return dict(methods=result, paired=comparisons,
                primary_holm_p=holm_primary(comparisons,primary_comparisons(methods)),
                paired_excluding_api_failures=paired(rows, methods, count,
                                                      include_api_failures=False),
                paired_policy='API and policy failures count as losses for end-to-end success. '
                              'Incomplete infrastructure failures remain visible and unpaired.')


def run(args):
    plan = load_plan(args.plan, args.phase)
    pairs = plan['seed_pairs']
    configs = config_matrix(args.direct_model)
    methods = list(configs)
    config_hash = digest(configs)
    runtime_version = code_version()
    if (args.phase=='pilot' and not args.fixture_for_tests and
            plan.get('direct_model',args.direct_model)!=args.direct_model):
        raise ValueError('The direct model must match the frozen pilot plan')
    if args.phase == 'holdout':
        if args.fixture_for_tests:
            raise ValueError('Holdout seed plans are reserved for the live run')
        if (plan.get('direct_model') != args.direct_model or plan.get('methods') != methods or
                plan.get('decision_config_sha256') != config_hash or
                plan.get('decision_runtime_version') != runtime_version or
                not plan.get('pilot_sha256')):
            raise ValueError('The live model and methods must match the frozen holdout plan')
    if args.fixture_for_tests:
        pairs = [FIXTURE_SEEDS]
    else:
        for config in configs.values():
            if config.get('max_llm_calls_per_stage') != 0:
                _key(config)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    bundle = root.parent/(root.name+'.tar.gz')
    if bundle.exists():
        raise FileExistsError(bundle)
    report = dict(schema='llm-study-paired-v1', phase=args.phase, seed_plan=plan,
                  seed_plan_sha256=digest(plan), seed_pairs=pairs, methods=methods,
                  direct_model=args.direct_model, historical_prediction_model='M1',
                  matched_for_direct='D1_matched' if args.direct_model=='M1' else 'D1_selected',
                  duration_seconds=48., fixture_only=args.fixture_for_tests,
                  decision_clock='paused_simulation', row_count_planned=len(pairs)*len(methods),
                  started_utc=datetime.now(timezone.utc).isoformat(), rows=[], error=None,
                  configs={name: wire_parameters(config) for name,config in configs.items()},
                  git_head=_git_head(),
                  decision_config_sha256=config_hash,
                  decision_runtime_version=runtime_version,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  maximum_live_calls=0 if args.fixture_for_tests else len(pairs)*(
                      2*(len(methods)-3)+48))
    emit(root/'comparison.json', report)
    try:
        for index, (first, second) in enumerate(pairs):
            order = methods[index % len(methods):] + methods[:index % len(methods)]
            for method in order:
                dir_ = root/f'{index:03d}-{method}'
                config_path = root/f'.{method}.json'
                emit(config_path, configs[method])
                run_episode_bundle(SimpleNamespace(
                    output_dir=dir_, bundle_path=root/(dir_.name+'.tar.gz'),
                    config=config_path, episode_id=f'pair-{index+1:03d}', first_seed=first,
                    second_seed=second, duration=48., port=args.port,
                    study_telemetry=True,
                    fixture_for_tests=args.fixture_for_tests and method != 'D0'))
                row = enrich(read_result(dir_, method, index, first, second), dir_)
                report['rows'].append(row)
                emit(root/'comparison.json', report)
                print(json.dumps(dict(index=index, method=method, status=row['status'],
                                      summary=row['summary'])), flush=True)
                if row['status'] not in ('completed','policy_failure','api_failure'):
                    raise RuntimeError(f'{dir_.name} had an infrastructure error')
    except (Exception, KeyboardInterrupt) as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        for method in methods:
            (root/f'.{method}.json').unlink(missing_ok=True)
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      aggregate=aggregate(report['rows'], methods, len(pairs)),
                      status='failed' if report['error'] else 'completed')
        emit(root/'comparison.json', report)
        comparison_bundle(root, bundle)
        print(json.dumps(dict(status=report['status'], upload_this=str(bundle),
                              attempts=len(report['rows']), error=report['error'])), flush=True)
    return 1 if report['error'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=PILOT_PLAN)
    parser.add_argument('--phase', choices=('pilot','holdout'), default='pilot')
    parser.add_argument('--direct-model', choices=MODELS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--fixture-for-tests', action='store_true')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_comparison(args.verify), indent=2))
        return 0
    if not args.output or not args.direct_model:
        parser.error('--output and --direct-model are required')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
