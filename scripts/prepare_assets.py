"""Fetch only the twelve pinned CC0 PBR maps, with SHA-256 verification.

No model calls, no robot meshes, no credentials. --check is entirely offline.
Archives are never extractall'ed; only whitelisted members are written.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
from arena_paths import PROJECT, workspace_root

MANIFEST = PROJECT / 'assets/pbr_manifest.json'
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


def inspect_assets(root):
    out = []
    for asset in json.loads(MANIFEST.read_text())['assets']:
        for item in asset['files']:
            path = root / asset['id'] / item['name']
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            out.append(dict(file=str(path), ok=actual == item['sha256'],
                            expected_sha256=item['sha256'], actual_sha256=actual))
    return out


def install_archive(asset, archive_bytes, root):
    """Verify all selected bytes before writing any; refuse altered existing files."""
    selected = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for item in asset['files']:
            if Path(item['name']).name != item['name'] or Path(asset['id']).name != asset['id']:
                raise ValueError('Invalid manifest path')
            info = archive.getinfo(item['name'])
            if info.file_size != item['size_bytes']:
                raise ValueError('Unexpected extracted size: ' + item['name'])
            data = archive.read(info)
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('SHA-256 mismatch: ' + item['name'])
            target = root / asset['id'] / item['name']
            if target.is_symlink() or target.parent.is_symlink():
                raise ValueError('Refusing symlink asset destination')
            if target.exists() and target.read_bytes() != data:
                raise FileExistsError('Preserving modified file; move it aside manually: ' + str(target))
            selected.append((target, data))
    for target, data in selected:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true', help='Check hashes without downloading')
    p.add_argument('--archive-dir', type=Path, help='Offline folder containing the four original ZIPs')
    p.add_argument('--assets-dir', type=Path, default=workspace_root() / 'simulators/assets/quadruped-arena/pbr')
    args = p.parse_args()
    if not args.check:
        for asset in json.loads(MANIFEST.read_text())['assets']:
            paths = [args.assets_dir / asset['id'] / item['name'] for item in asset['files']]
            if all(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256']
                   for p, item in zip(paths, asset['files'])):
                continue
            filename = f"{asset['id']}_{asset['resolution']}-JPG.zip"
            if args.archive_dir:
                archive = args.archive_dir / filename
                if not archive.exists():
                    archive = args.archive_dir / asset['id'] / filename
                data = archive.read_bytes()
            else:
                print('Downloading CC0 asset: ' + asset['download'], flush=True)
                req = urllib.request.Request(asset['download'], headers={'User-Agent': 'QuadrupedArena-reproducer/1.0'})
                with urllib.request.urlopen(req, timeout=120) as response:
                    data = response.read(MAX_ARCHIVE_BYTES + 1)
            if len(data) > MAX_ARCHIVE_BYTES:
                raise ValueError('Archive exceeds safety limit')
            install_archive(asset, data, args.assets_dir)
    checks = inspect_assets(args.assets_dir)
    print(json.dumps({'passed': sum(x['ok'] for x in checks), 'total': len(checks), 'checks': checks}, indent=2))
    return 0 if all(x['ok'] for x in checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
