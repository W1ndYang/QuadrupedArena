"""Paired, held-out simulation comparison; return all runs in one archive.

The fixed baseline was chosen on design seeds 810000/820000. Do not tune it
on the comparison seeds. No model is called for either fixed or numeric runs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import tarfile
import tempfile
import time
import traceback
from types import SimpleNamespace

from llm_bundle import create_bundle, verify_bundle
from llm_run_ubuntu import HERE, PROJECT, _git_head, _stamp, run as run_service
from llm_response_cache import add_recorded_events


METHODS = ('fixed_no_peek', 'fixed_two_peek', 'numeric', 'llm')
DESIGN_SEEDS = (810000, 820000)
TRACK_ABSTENTIONS = {
    'No associated two-sample guard track after observation cap',
    'Boundary reassessment has no public guard track',
    'Boundary guard track unavailable and recovery exhausted',
}


class FixedPolicy:
    """One frozen script: north/zero wait/no smoke in both stages.

    This was the fastest successful design-case candidate. Gate boundaries
    always continue. Only the public stage and legal-action mask are used.
    """
    def __init__(self, episode_id, duration, peeks):
        from llm_option_policy import LLMOptionPolicy
        self.observer = LLMOptionPolicy(episode_id, None, duration)
        self.required_peeks = peeks
        self.stage = -1
        self.peeks = 0

    def observe(self, observation, mask):
        self.observer.observe(observation, mask)

    def __call__(self, observation, mask):
        stage = self.observer.stage
        if stage != self.stage:
            self.stage = stage
            self.peeks = 0
        if mask[7]:
            action = 7
        elif self.peeks < self.required_peeks:
            action = (2, 3)[self.peeks]
            self.peeks += 1
        else:
            action = 6
        if not mask[action]:
            raise ValueError(f'Frozen baseline selected illegal action {action}')
        return action, 0., 0.


def run_fixed(method, first_seed, second_seed, directory, duration):
    from online_env import ACTION_NAMES
    from option_env import run_episode
    directory.mkdir(parents=True)
    episode_id = directory.name
    policy = FixedPolicy(episode_id, duration, 2 if method == 'fixed_two_peek' else 0)
    start = time.perf_counter()
    result = dict(episode_id=episode_id, first_seed=first_seed, second_seed=second_seed,
                  started_utc=datetime.now(timezone.utc).isoformat())
    try:
        summary, records = run_episode(first_seed, second_seed, policy, duration)
        result.update(status='completed', summary=summary, options=[
            dict(stage=row['stage'], action=ACTION_NAMES[row['action']],
                 start=row['time'], end=row['end_time'],
                 duration_seconds=row['elapsed_seconds'],
                 termination=row['termination_reason']) for row in records])
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    result.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                  own_trajectory=policy.observer.own_trajectory,
                  public_guard_observations=policy.observer.samples)
    (directory / f'{episode_id}.json').write_text(json.dumps(result, indent=2)+'\n')
    manifest = dict(schema='llm-run-manifest-v1', episode_id=episode_id,
                    first_seed=first_seed, second_seed=second_seed,
                    status=result['status'], decision_mode=method, fixture_only=False,
                    git_head=_git_head(), llm_inference_attempts=0,
                    llm_responses_received=0, completed_model_wall_seconds=0.,
                    wall_seconds=time.perf_counter()-start,
                    # Fixed scripts are formally simulated, but have no service to replay.
                    replay_matches=None)
    (directory / 'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def read_result(directory, method, index, first_seed, second_seed):
    manifest = json.loads((directory / 'run_manifest.json').read_text())
    episode_path = directory / f"{manifest['episode_id']}.json"
    episode = json.loads(episode_path.read_text()) if episode_path.exists() else {}
    events_path = directory / 'decision_events.jsonl'
    events = ([json.loads(line) for line in events_path.read_text().splitlines()]
              if events_path.exists() else [])
    decisions = [event for event in events if event['kind'] == 'decision']
    abstention = next((event for event in reversed(decisions)
                      if event['response'].get('status') == 'error' and
                      (event['response'].get('error_type') in ('PolicyAbstention','ModelOutputError') or
                       event['response'].get('error') in TRACK_ABSTENTIONS)), None)
    api_seconds, local_seconds, model_seconds, reuse_seconds = [], [], [], []
    for event in decisions:
        response = event['response']
        raw = (event.get('llm_response') or
               response.get('reasoning', {}).get('raw_response'))
        # A received response still consumed model time if validation failed.
        if raw is not None:
            if raw.get('response_source') == 'recorded_exact_request':
                reuse_seconds.append(event['service_wall_seconds'])
            else:
                api_seconds.append(event['service_wall_seconds'])
                model_seconds.append(raw.get('wall_seconds',
                    response.get('timing', {}).get('api_seconds', 0.)))
        elif response['status'] in ('plan', 'boundary'):
            local_seconds.append(event['service_wall_seconds'])
    summary = episode.get('summary')
    status = manifest['status']
    if abstention is not None:
        # It is the policy's inability to decide under its own observation cap,
        # not a machine failure or an ignorable missing trial. Count it as a loss.
        status = 'policy_failure'
        model_error=abstention['response'].get('error_type')=='ModelOutputError'
        summary = dict(winner='defender', reason='model_output_invalid' if model_error else 'policy_abstention',
                       seconds=abstention['request']['time'],
                       synthetic_result=True, formal_terminal_reached=False)
    return dict(index=index, method=method, first_seed=first_seed, second_seed=second_seed,
                status=status, summary=summary,
                failure_category=('model_output' if abstention and
                    abstention['response'].get('error_type')=='ModelOutputError' else
                    'information_budget' if abstention else None),
                replay_matches=manifest.get('replay_matches'),
                error=manifest.get('error') or episode.get('error'),
                llm_inference_attempts=manifest.get('llm_inference_attempts', 0),
                llm_responses_received=manifest.get('llm_responses_received', 0),
                recorded_responses_reused=manifest.get('recorded_responses_reused', 0),
                fresh_model_responses_received=manifest.get('fresh_model_responses_received',
                    manifest.get('llm_responses_received', 0)),
                model_seconds=model_seconds, api_service_seconds=api_seconds,
                recorded_reuse_service_seconds=reuse_seconds,
                model_validation_events=[dict(time=event['request']['time'],
                    **event['response']['reasoning']['validation']) for event in decisions
                    if event['response'].get('reasoning',{}).get('validation') is not None],
                local_service_seconds=local_seconds,
                evaluation=episode.get('service_evaluation', {}).get('evaluation'))


def summarize(rows, methods, count):
    result = {}
    for method in methods:
        attempted = [row for row in rows if row['method'] == method]
        completed = [row for row in attempted if row['status'] == 'completed']
        evaluated = [row for row in attempted if row['status'] in ('completed', 'policy_failure')]
        wins = [row for row in completed if row['summary']['winner'] == 'attacker']
        result[method] = dict(planned=count, attempted=len(attempted),
            completed=len(completed), evaluated=len(evaluated),
            policy_failures=len(evaluated)-len(completed),
            model_output_failures=sum(row.get('failure_category')=='model_output' for row in evaluated),
            partially_rejected_model_responses=sum(bool(v['rejected_hypotheses']) for row in attempted
                                                   for v in row.get('model_validation_events',[])),
            run_errors=len(attempted)-len(evaluated),
            unattempted=count-len(attempted), wins=len(wins),
            win_rate_on_evaluated=len(wins)/len(evaluated) if evaluated else None,
            success_rate_per_attempted=len(wins)/len(attempted) if attempted else None,
            mean_winning_game_seconds=(sum(row['summary']['seconds'] for row in wins)/len(wins)
                                       if wins else None),
            llm_inference_attempts=sum(row['llm_inference_attempts'] for row in attempted),
            recorded_responses_reused=sum(row.get('recorded_responses_reused', 0) for row in attempted),
            fresh_model_responses_received=sum(row.get('fresh_model_responses_received', 0) for row in attempted),
            model_seconds=[x for row in attempted for x in row['model_seconds']],
            api_service_seconds=[x for row in attempted for x in row['api_service_seconds']],
            local_service_seconds=[x for row in attempted for x in row['local_service_seconds']])
    pairs = {}
    if 'llm' in methods:
        for method in methods:
            if method == 'llm':
                continue
            lookup = {(row['index'], row['method']): row for row in rows}
            compared = both_won = llm_only = baseline_only = both_lost = 0
            for index in range(count):
                a, b = lookup.get((index, 'llm')), lookup.get((index, method))
                if (not a or not b or a['status'] not in ('completed', 'policy_failure') or
                        b['status'] not in ('completed', 'policy_failure')):
                    continue
                aw, bw = a['summary']['winner'] == 'attacker', b['summary']['winner'] == 'attacker'
                compared += 1
                both_won += aw and bw
                llm_only += aw and not bw
                baseline_only += bw and not aw
                both_lost += not aw and not bw
            n = llm_only + baseline_only
            # Exact two-sided McNemar/binomial test, conditional on discordant pairs.
            p = min(1., 2.*sum(math.comb(n, k) for k in
                    range(min(llm_only, baseline_only)+1))/2**n) if n else 1.
            pairs[method] = dict(completed_pairs=compared, excluded_pairs=count-compared,
                both_won=both_won, llm_only_won=llm_only, baseline_only_won=baseline_only,
                both_lost=both_lost, exact_two_sided_p=p)
    return dict(methods=result, paired_vs_llm=pairs,
                note='Small screening experiment; no automatic feasibility verdict. '
                     'Observation-budget abstentions and unusable model outputs count as policy losses; run errors and '
                     'unattempted trials remain explicit. Success per attempted includes run errors.')


def comparison_bundle(root, output):
    files = [root / 'comparison.json', *sorted(root.glob('*.tar.gz'))]
    for path in files[1:]:
        verify_bundle(path)
    rows = [dict(path=path.name, size_bytes=path.stat().st_size,
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files]
    contents = [(path.name, path.read_bytes()) for path in files]
    contents.append(('archive_manifest.json', json.dumps(dict(
        schema='llm-comparison-bundle-v1', files=rows), indent=2).encode()))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(output, 'x:gz') as archive:
            for name, data in contents:
                entry = tarfile.TarInfo(f'llm_comparison/{name}')
                entry.size, entry.mode = len(data), 0o600
                archive.addfile(entry, io.BytesIO(data))
        verify_comparison(output)
    except Exception:
        output.unlink(missing_ok=True)
        raise


def verify_comparison(path):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if (len(set(names)) != len(names) or any(not m.isfile() or
                not m.name.startswith('llm_comparison/') or
                '/' in m.name[len('llm_comparison/'):] for m in members)):
            raise ValueError('Unexpected comparison archive member')
        manifest = json.load(archive.extractfile('llm_comparison/archive_manifest.json'))
        expected = {'llm_comparison/archive_manifest.json'}
        for row in manifest['files']:
            name = 'llm_comparison/' + row['path']
            expected.add(name)
            data = archive.extractfile(name).read()
            if len(data) != row['size_bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError(f'Checksum mismatch: {name}')
        if expected != set(names):
            raise ValueError('Comparison member list differs from manifest')
    return manifest


def recorded_cache_from_comparison(path):
    """Validate both archive levels before importing public model exchanges."""
    verify_comparison(path)
    entries = {}
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with tarfile.open(path, 'r:gz') as archive, tempfile.TemporaryDirectory() as directory:
        for member in archive.getmembers():
            if not member.name.endswith('.tar.gz'):
                continue
            data = archive.extractfile(member).read()
            inner_path = Path(directory) / 'run.tar.gz'
            inner_path.write_bytes(data)
            verify_bundle(inner_path)
            with tarfile.open(inner_path, 'r:gz') as inner:
                if 'llm_run/decision_events.jsonl' not in inner.getnames():
                    continue
                events = [json.loads(line) for line in inner.extractfile(
                    'llm_run/decision_events.jsonl').read().decode().splitlines()]
                add_recorded_events(entries, events, dict(source_archive_sha256=source_sha,
                    inner_bundle=member.name, inner_bundle_sha256=hashlib.sha256(data).hexdigest()))
    return dict(schema='llm-exact-response-cache-v1', entries=entries,
                source_archive_sha256=source_sha)


def run(args):
    methods = args.methods
    config = json.loads(args.config.read_text())
    if 'llm' in methods and config.get('max_llm_calls_per_stage') != 1:
        raise ValueError('Comparison requires max_llm_calls_per_stage=1 (at most 2 calls/episode)')
    pairs = [(args.first_seed+i*args.first_stride, args.second_seed+i*args.second_stride)
             for i in range(args.count)]
    if any(a == DESIGN_SEEDS[0] or b == DESIGN_SEEDS[1] for a, b in pairs):
        raise ValueError('Comparison must exclude the design seeds 810000/820000')
    root = args.output_dir.resolve()
    output = args.bundle_path.resolve() if args.bundle_path else root.with_suffix('.tar.gz')
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {root}')
    if output.exists() or output == root or root in output.parents:
        raise ValueError('Use a new bundle path outside the output directory')
    root.mkdir(parents=True, exist_ok=True)
    report = dict(schema='llm-paired-comparison-v1', git_head=_git_head(),
        fixture_only=args.fixture_for_tests, started_utc=datetime.now(timezone.utc).isoformat(),
        methods=methods, seed_pairs=pairs, count=args.count, duration=args.duration,
        maximum_live_api_calls=(2*args.count if 'llm' in methods and not args.fixture_for_tests else 0),
        fixed_baseline=dict(design_seeds=DESIGN_SEEDS, stage0='north/0/no smoke',
            stage1='north/0/no smoke', gate='always continue',
            observation='0 or 2 peeks per stage; south then north',
            design_archive='ubuntu-20261118-073656-9a7a8d.tar.gz'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=[], error=None)
    report['boundary_track_recovery'] = config.get('boundary_track_recovery', False)
    numeric_path = root / '.numeric-runtime.json'
    numeric_path.write_text(json.dumps(dict(config, max_llm_calls_per_stage=0)))
    llm_path = args.config
    llm_runtime_path = root / '.llm-runtime.json'
    cache_path = root / '.recorded-response-cache.json'
    secret_values = [os.environ.get(config.get('api_key_env', 'DEEPSEEK_API_KEY'))]
    try:
        if args.reuse_responses_from:
            cached = recorded_cache_from_comparison(args.reuse_responses_from)
            cache_path.write_text(json.dumps(cached, ensure_ascii=False))
            llm_runtime_path.write_text(json.dumps(dict(config,
                recorded_response_cache_file=str(cache_path))))
            llm_path = llm_runtime_path
            report['recorded_response_reuse'] = dict(source=str(args.reuse_responses_from),
                source_archive_sha256=cached['source_archive_sha256'], entries=len(cached['entries']),
                contract='Exact API configuration and complete message equality; misses call live API. '
                         'All episodes rerun under current policy; old outcomes are not pooled.')
        if config.get('api_key_file'):
            secret_values.append(Path(config['api_key_file']).expanduser().read_text().strip())
        for index, (first_seed, second_seed) in enumerate(pairs):
            # Rotate order to avoid always giving a method the same position in time.
            order = methods[index % len(methods):] + methods[:index % len(methods)]
            for method in order:
                directory = root / f'{index:03d}-{method}'
                archive = root / f'{directory.name}.tar.gz'
                if method.startswith('fixed_'):
                    run_fixed(method, first_seed, second_seed, directory, args.duration)
                    create_bundle(directory, archive, secret_values)
                else:
                    run_service(SimpleNamespace(output_dir=directory, bundle_path=archive,
                        config=numeric_path if method == 'numeric' else llm_path,
                        episode_id=directory.name, first_seed=first_seed, second_seed=second_seed,
                        duration=args.duration, port=args.port,
                        fixture_for_tests=args.fixture_for_tests and method == 'llm'))
                row = read_result(directory, method, index, first_seed, second_seed)
                report['rows'].append(row)
                print(json.dumps(dict(index=index, method=method, status=row['status'],
                                      summary=row['summary'])), flush=True)
                (root / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
                if row['status'] not in ('completed', 'policy_failure'):
                    raise RuntimeError(f'{directory.name} failed; stopped without retrying model calls')
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        numeric_path.unlink(missing_ok=True)
        llm_runtime_path.unlink(missing_ok=True)
        cache_path.unlink(missing_ok=True)
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      status='failed' if report['error'] else 'completed',
                      aggregate=summarize(report['rows'], methods, args.count))
        (root / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
        comparison_bundle(root, output)
        print(json.dumps(dict(status=report['status'], upload_this=str(output),
                              aggregate=report['aggregate'], error=report['error'])), flush=True)
    return 1 if report['error'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'config/llm_decision_deepseek.json')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--first-seed', type=int, default=810100)
    parser.add_argument('--second-seed', type=int, default=820100)
    parser.add_argument('--first-stride', type=int, default=17)
    parser.add_argument('--second-stride', type=int, default=19)
    parser.add_argument('--duration', type=float, default=48.)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--fixture-for-tests', action='store_true')
    parser.add_argument('--reuse-responses-from', type=Path,
                        help='Verified comparison archive; reuse only identical saved model requests')
    parser.add_argument('--output-dir', type=Path, default=PROJECT/
        f'.arena-workspace/experiments/llm-motion/comparison-{_stamp()}-{secrets.token_hex(3)}')
    parser.add_argument('--bundle-path', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        manifest = verify_comparison(args.verify)
        print(json.dumps(dict(status='verified', files=len(manifest['files']))))
        return
    if (args.count <= 0 or args.count > 1000 or args.first_stride <= 0 or
            args.second_stride <= 0 or len(set(args.methods)) != len(args.methods) or
            not math.isfinite(args.duration) or args.duration <= 0.):
        parser.error('Use count 1..1000, positive strides/duration, and unique methods')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
