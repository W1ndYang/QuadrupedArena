"""Generate a complete, replay-compatible deterministic baseline without GPU/API."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from adaptive_motion import build, load_config
from arena_paths import PROJECT, workspace_root
from replay_io import load_replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=PROJECT / 'configs/trial_rotating.json')
    p.add_argument('--output', type=Path)
    p.add_argument('--seed', type=int, help='Override seed; the unmodified baseline uses 20261021')
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg['seed'] = args.seed
    root = args.output or workspace_root() / 'experiments/runs/quadruped-arena' / (
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-cpu')
    # Never overwrite a previous experiment.
    root.mkdir(parents=True, exist_ok=False)
    states, events, validation = build(cfg)
    for name, value in [('trial_config.json', cfg), ('timeline.json', states),
                        ('events.json', events), ('validation.json', validation)]:
        (root / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    load_replay(root)
    result = dict(status='passed', kind='cpu_kinematic_baseline', training=False,
                  dynamics_validated=False, llm_calls=0, seed=cfg['seed'],
                  simulation_seconds=states[-1]['t'], ticks=len(states),
                  validation=validation,
                  hashes={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ('trial_config.json', 'timeline.json', 'events.json', 'validation.json')})
    (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(directory=str(root.resolve()), **result), indent=2))


if __name__ == '__main__':
    main()
