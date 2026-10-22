"""Read-only environment check. Never reads credentials or contacts model APIs."""
import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from arena_paths import workspace_root, dependency_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--isaac', action='store_true', help='Also require renderer dependencies, GPU and robot files')
    args = p.parse_args()
    checks = {'python_3_11': {'ok': sys.version_info[:2] == (3, 11), 'value': sys.version.split()[0]}}
    names = ['numpy', 'scipy', 'pillow', 'matplotlib', 'trimesh', 'imageio', 'imageio-ffmpeg']
    if args.isaac:
        names += ['isaacsim', 'isaaclab', 'torch']
    for name in names:
        try:
            checks[name] = {'ok': True, 'version': importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            checks[name] = {'ok': False, 'error': 'not installed'}
    if args.isaac:
        available = shutil.which('nvidia-smi')
        probe = subprocess.run([available, '--query-gpu=name,memory.total,driver_version', '--format=csv'],
                               capture_output=True, text=True, timeout=20) if available else None
        checks['gpu'] = {'ok': probe is not None and probe.returncode == 0,
                         'value': probe.stdout.strip() if probe else 'nvidia-smi not found'}
        repo = dependency_path('unitree_ros')
        for model in ['go2_description/urdf/go2_description.urdf',
                      'go2w_description/urdf/go2w_description.urdf', 'g1_description/g1_23dof.urdf']:
            checks[model] = {'ok': (repo / 'robots' / model).is_file()}
        from prepare_assets import inspect_assets
        assets = inspect_assets(workspace_root() / 'simulators/assets/quadruped-arena/pbr')
        checks['pbr'] = {'ok': all(x['ok'] for x in assets), 'files': assets}
    print(json.dumps({'workspace': str(workspace_root()), 'checks': checks}, indent=2))
    return 0 if all(x['ok'] for x in checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
