"""Portable source, runtime and credential locations (no I/O at import)."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def workspace_root():
    """Keep generated assets/runs out of Git; never infer from clone depth."""
    return Path(os.environ.get('ARENA_WORKSPACE', PROJECT / '.arena-workspace')).expanduser().resolve()


def dependency_path(name):
    env = {'IsaacLab': 'ISAACLAB_PATH', 'unitree_ros': 'UNITREE_ROS_DIR',
           'unitree_rl_lab': 'UNITREE_RL_LAB_PATH'}[name]
    return Path(os.environ.get(env, workspace_root() / 'projects/unitree' / name)).expanduser().resolve()


def credential_path(filename):
    base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')).expanduser()
    return base / 'quadruped-arena' / filename


def rules_reference_path():
    return PROJECT / 'configs/rules_reference_v04.json'
