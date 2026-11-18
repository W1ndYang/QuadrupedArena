"""Run one independent-service episode, replay it, and package all logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import subprocess
import sys
import time

from llm_bundle import create_bundle


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def _stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')


def _git_head():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _wait_for_service(process, log_path, port, timeout=30.):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if process.poll() is not None:
            raise RuntimeError(f'Decision service exited with code {process.returncode}; '
                               f'see {log_path}')
        try:
            lines = log_path.read_text().splitlines()
            ready = any(json.loads(line).get('status') == 'ready' for line in lines
                        if line.startswith('{'))
            if ready:
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1.)
                try:
                    conn.request('GET', '/health')
                    response = conn.getresponse()
                    health = json.loads(response.read())
                finally:
                    conn.close()
                if response.status == 200 and health.get('status') == 'ready':
                    return health
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(.1)
    raise TimeoutError(f'Decision service did not become ready; see {log_path}')


def _run_logged(command, log_path):
    with log_path.open('wb') as output:
        return subprocess.run(command, cwd=PROJECT, stdout=output,
                              stderr=subprocess.STDOUT, check=False).returncode


def run(args):
    run_dir = args.output_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Run directory must be empty: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    archive = (args.bundle_path.resolve() if args.bundle_path else
               run_dir.parent / f'{run_dir.name}.tar.gz')
    if archive.exists():
        raise FileExistsError(f'Bundle already exists: {archive}')

    config = json.loads(args.config.read_text())
    safe_config = {key: value for key, value in config.items()
                   if key not in ('api_key', 'api_key_file')}
    (run_dir / 'config_effective.json').write_text(
        json.dumps(safe_config, ensure_ascii=False, indent=2) + '\n')
    shutil.copyfile(HERE / 'llm_map_manifest.json', run_dir / 'map_manifest.json')
    shutil.copyfile(HERE / 'llm_guard_rules.json', run_dir / 'guard_rules.json')

    secret_values = [os.environ.get(name) for name in
                     ('DEEPSEEK_API_KEY', 'QWEN_API_KEY')]
    key_file = config.get('api_key_file')
    if key_file:
        try:
            secret_values.append(Path(key_file).expanduser().read_text().strip())
        except OSError:
            pass

    episode_id = args.episode_id or f'ep-{secrets.token_hex(8)}'
    manifest = dict(schema='llm-run-manifest-v1',
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    episode_id=episode_id, first_seed=args.first_seed,
                    second_seed=args.second_seed, duration_seconds=args.duration,
                    provider=config.get('provider'), model=config.get('model'),
                    fixture_only=args.fixture_for_tests, git_head=_git_head(),
                    decision_mode=('fixture_debug' if args.fixture_for_tests else
                        'numeric_only' if config.get('max_llm_calls_per_stage')==0 else
                        'llm_with_local_updates'),
                    source_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                   for name in ('llm_run_ubuntu.py', 'llm_bundle.py',
                                                'llm_run_episode.py', 'llm_replay.py')},
                    python=platform.python_version(), service_health=None,
                    map_review_status=None, results_provisional=None,
                    episode_exit_code=None, replay_exit_code=None,
                    replay_matches=None, error=None)
    service = None
    service_log = run_dir / 'service_stdout.log'
    try:
        with service_log.open('wb') as output:
            command = [sys.executable, str(HERE / 'llm_decision_service.py'),
                       '--config', str(args.config.resolve()), '--log-dir', str(run_dir),
                       '--port', str(args.port)]
            if args.fixture_for_tests:
                command.append('--fixture-for-tests')
            service = subprocess.Popen(command, cwd=PROJECT, stdout=output,
                                       stderr=subprocess.STDOUT)
            manifest['service_health'] = _wait_for_service(service, service_log, args.port)
            manifest['map_review_status'] = manifest['service_health'].get('map_review_status')
            manifest['results_provisional'] = (
                manifest['map_review_status'] != 'human_approved')
            episode_command = [sys.executable, str(HERE / 'llm_run_episode.py'),
                               '--service-url', f'http://127.0.0.1:{args.port}',
                               '--first-seed', str(args.first_seed),
                               '--second-seed', str(args.second_seed),
                               '--duration', str(args.duration),
                               '--episode-id', episode_id,
                               '--output-dir', str(run_dir)]
            manifest['episode_exit_code'] = _run_logged(
                episode_command, run_dir / 'episode_stdout.log')
    except Exception as exc:
        manifest['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if service is not None and service.poll() is None:
            service.terminate()
            try:
                service.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait(timeout=5.)

    if (run_dir / 'decision_events.jsonl').exists():
        try:
            events=[json.loads(line) for line in
                    (run_dir/'decision_events.jsonl').read_text().splitlines() if line.strip()]
            decision_events=[row for row in events if row['kind']=='decision']
            decisions=list({(row['request']['episode_id'],row['request']['request_id']):row
                            for row in decision_events}.values())
            manifest['decision_events']=len(decision_events)
            manifest['unique_decision_requests']=len(decisions)
            manifest['llm_inference_attempts']=sum(bool(row.get('llm_request')) for row in decisions)
            raw=[row.get('llm_response') or row['response'].get('reasoning',{}).get('raw_response')
                 for row in decisions]
            manifest['llm_responses_received']=sum(row is not None for row in raw)
            manifest['recorded_responses_reused']=sum(
                row is not None and row.get('response_source')=='recorded_exact_request'
                for row in raw)
            manifest['fixture_responses_received']=(manifest['llm_responses_received']
                                                    if args.fixture_for_tests else 0)
            manifest['fresh_model_responses_received']=(manifest['llm_responses_received']-
                manifest['recorded_responses_reused']-manifest['fixture_responses_received'])
            manifest['completed_model_wall_seconds']=sum(float(row.get('wall_seconds') or 0.)
                                                         for row in raw if row is not None)
            manifest['local_boundary_decisions']=sum(row['response']['status']=='boundary'
                                                      for row in decisions)
            replay_command = [sys.executable, str(HERE / 'llm_replay.py'),
                              '--input-log', str(run_dir / 'decision_events.jsonl'),
                              '--output', str(run_dir / 'replay.json'),
                              '--config', str(args.config.resolve())]
            manifest['replay_exit_code'] = _run_logged(replay_command,
                                                       run_dir / 'replay_stdout.log')
            if manifest['replay_exit_code'] == 0:
                replay = json.loads((run_dir / 'replay.json').read_text())
                manifest['replay_matches'] = (
                    replay['count'] > 0 and
                    len(replay['finishes']) == 1 and
                    replay['count'] == replay['same_decisions'] and
                    all(row['same_evaluation'] for row in replay['finishes']))
        except Exception as exc:
            manifest['error'] = f'{type(exc).__name__}: {exc}'

    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['status'] = ('completed' if manifest['error'] is None and
                          manifest['episode_exit_code'] == 0 and
                          manifest['replay_exit_code'] == 0 and
                          manifest['replay_matches'] else 'failed')
    (run_dir / 'run_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    result = create_bundle(run_dir, archive, secret_values)
    print(json.dumps(dict(status=manifest['status'],
                          episode_id=episode_id, upload_this=result['archive'],
                          archive_sha256=result['archive_sha256'],
                          included_files=result['file_count'],
                          error=manifest['error']), ensure_ascii=False), flush=True)
    return 0 if manifest['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path,
                        default=HERE / 'config/llm_decision_deepseek.json')
    parser.add_argument('--output-dir', type=Path,
                        default=PROJECT / '.arena-workspace/experiments/llm-motion' /
                        f'ubuntu-{_stamp()}-{secrets.token_hex(3)}')
    parser.add_argument('--bundle-path', type=Path)
    parser.add_argument('--first-seed', type=int, default=810000)
    parser.add_argument('--second-seed', type=int, default=820000)
    parser.add_argument('--duration', type=float, default=48.)
    parser.add_argument('--episode-id')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--fixture-for-tests', action='store_true',
                        help='Local engineering test only; no LLM API call')
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
