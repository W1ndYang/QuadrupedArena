"""Package one LLM decision run into a single, checkable archive."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


KNOWN_FILES = {
    'decision_events.jsonl', 'service_stdout.log', 'episode_stdout.log',
    'replay_stdout.log', 'replay.json', 'run_manifest.json',
    'config_effective.json', 'map_manifest.json', 'guard_rules.json',
}


def _episode_json(path):
    try:
        value = json.loads(path.read_text())
    except (UnicodeError, ValueError):
        return False
    return (isinstance(value, dict) and
            isinstance(value.get('episode_id'), str) and
            value.get('status') in ('completed', 'failed'))


def _included_files(run_dir, additional_files=()):
    files = []
    for path in sorted(run_dir.iterdir()):
        if path.is_symlink():
            raise ValueError(f'Symlinks are not allowed in a run directory: {path.name}')
        if not path.is_file():
            continue
        if path.name in KNOWN_FILES or path.name in additional_files or (
                path.suffix == '.json' and _episode_json(path)):
            files.append(path)
    if not files:
        raise ValueError('No recognized run files to package')
    return files


def create_bundle(run_dir, archive_path, secret_values=(), *, additional_files=()):
    run_dir = Path(run_dir).resolve()
    archive_path = Path(archive_path).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f'Run directory does not exist: {run_dir}')
    if archive_path.parent == run_dir or run_dir in archive_path.parents:
        raise ValueError('Archive must be outside the run directory')
    if archive_path.exists():
        raise FileExistsError(f'Archive already exists: {archive_path}')
    files = _included_files(run_dir, additional_files)
    secrets = [value.encode() for value in secret_values if isinstance(value, str) and value]
    rows = []
    contents = []
    for path in files:
        data = path.read_bytes()
        if any(secret in data for secret in secrets):
            raise ValueError(f'Credential value found in {path.name}; archive not created')
        rows.append(dict(path=path.name, size_bytes=len(data),
                         sha256=hashlib.sha256(data).hexdigest()))
        contents.append((path.name, data))
    manifest = dict(schema='llm-run-bundle-v1',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    run_name=run_dir.name, files=rows)
    contents.append(('archive_manifest.json',
                     (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode()))
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, 'x:gz') as archive:
            for name, data in contents:
                entry = tarfile.TarInfo(f'llm_run/{name}')
                entry.size = len(data)
                entry.mode = 0o600
                archive.addfile(entry, io.BytesIO(data))
        verify_bundle(archive_path)
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return dict(archive=str(archive_path),
                archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                file_count=len(rows), size_bytes=archive_path.stat().st_size)


def verify_bundle(archive_path):
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        if any(not member.isfile() or not member.name.startswith('llm_run/')
               or '/' in member.name[len('llm_run/'):] for member in members):
            raise ValueError('Unexpected archive member')
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member')
        manifest = json.load(archive.extractfile('llm_run/archive_manifest.json'))
        expected = {'llm_run/archive_manifest.json'}
        for row in manifest['files']:
            name = f"llm_run/{row['path']}"
            expected.add(name)
            data = archive.extractfile(name).read()
            if len(data) != row['size_bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError(f'Checksum mismatch: {name}')
        if set(names) != expected:
            raise ValueError('Archive member list differs from manifest')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path,
                        help='Optional config to scan an external api_key_file value')
    parser.add_argument('--verify', type=Path,
                        help='Verify a returned archive without extracting it')
    args = parser.parse_args()
    if args.verify:
        if args.run_dir or args.output or args.config:
            parser.error('--verify cannot be combined with creation options')
        manifest = verify_bundle(args.verify)
        print(json.dumps(dict(status='verified',archive=str(args.verify),
                              file_count=len(manifest['files']),
                              run_name=manifest['run_name']),ensure_ascii=False))
        return
    if not args.run_dir or not args.output:
        parser.error('--run-dir and --output are required when creating a bundle')
    values = [os.environ.get(name) for name in ('DEEPSEEK_API_KEY', 'QWEN_API_KEY')]
    if args.config:
        config = json.loads(args.config.read_text())
        if config.get('api_key_file'):
            values.append(Path(config['api_key_file']).expanduser().read_text().strip())
    result = create_bundle(args.run_dir, args.output, values)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
