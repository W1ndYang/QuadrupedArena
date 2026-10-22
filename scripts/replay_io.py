"""Strict read-only loader for a completed, uniformly sampled referee replay."""
import hashlib
import json
from pathlib import Path


def load_replay(directory):
    directory=Path(directory).resolve()
    names=('trial_config.json','timeline.json','events.json','validation.json')
    blobs={name:(directory/name).read_bytes() for name in names}
    cfg,states,events,validation=(json.loads(blobs[name]) for name in names)
    if not states or not states[-1].get('finished'):
        raise ValueError('Only finished matches can be rendered as completed replays')
    for index,state in enumerate(states):
        if abs(state['t']-index*cfg['dt'])>1e-6:
            raise ValueError('Replay must start at zero and contain every simulation tick')
    if abs(states[-1]['score']-validation['score'])>1e-6:
        raise ValueError('Replay score differs from validation record')
    manifest=dict(source=str(directory),policy_rerun=False,
        source_sha256={name:hashlib.sha256(blob).hexdigest() for name,blob in blobs.items()})
    return cfg,states,events,validation,manifest
