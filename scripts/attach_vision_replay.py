"""Attach a validated post-match native Isaac replay to its source viewer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from llm_trial import dump
from vision_report import build


def attach(source, render):
    source, render = source.resolve(), render.resolve()
    original = json.loads((source / 'result.json').read_text())
    result = json.loads((render / 'result.json').read_text())
    assert original['status'] == result['status'] == 'passed'
    assert Path(result['replay']['source']).resolve() == source
    assert result['replay']['policy_rerun'] is False
    assert result['match_seconds'] == original['simulation_seconds']
    assert result['validation']['score'] == original['score']
    for name, digest in result['replay']['source_sha256'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    files = dict(gif='demo.gif', mp4='demo.mp4', poster='poster.png')
    assert all((render / name).is_file() for name in files.values())
    dump(source / 'third_person_replay.json', dict(
        status='passed', renderer='Isaac ' + result['renderer'],
        source_run=source.name, render_run=render.name,
        **{key: os.path.relpath(render / name, source) for key, name in files.items()},
        duration_seconds=result['duration_seconds'], match_seconds=result['match_seconds'],
        score=original['score'], policy_rerun=False, new_api_calls=0,
        note='Post-match full-scene presentation only, never planner input. Includes intro/outro and cinematic collapse when reached.'))
    build(source)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('render', type=Path)
    a = p.parse_args()
    attach(a.source, a.render)
