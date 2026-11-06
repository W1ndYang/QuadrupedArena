"""Verify the viewing package offline, without dependencies, Isaac or API access."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    manifest = read(ROOT / 'manifest.json')
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    provenance = read(ROOT / 'provenance.json')
    for name, expected in provenance['unchanged_text_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    js = (ROOT / 'data.js').read_text(encoding='utf-8')
    bundle = json.loads(js.removeprefix('window.ARENA_SHOWCASE = ').removesuffix(';\n'))
    decisions = tokens = 0
    for game in bundle['games']:
        folder = ROOT / f"run_{game['case']['run']}"
        ds = read(folder / 'decisions.json')
        assert ds == game['decisions']
        assert read(folder / 'result.json') == game['result']
        calls = read(folder / 'call_records.json')
        assert len(ds) == len(calls) == len(game['steps'])
        for d, call, step in zip(ds, calls, game['steps']):
            raw = json.loads(call['content'])
            assert raw == {k:d[k] for k in ('observation_id','skill','params','assessment')}
            assert d['simulation_seconds'] == step['start']
            assert call['usage'] == step['usage']
            assert (ROOT / step['image']).is_file()
            if step['routeplot']:
                assert (ROOT / step['routeplot']).is_file()
            decisions += 1
            tokens += call['usage']['total_tokens']
        media = read(folder / 'media.json')
        assert media['frames'] == media['source_render']['gif_frames']
        assert abs(media['duration_seconds'] - game['result']['seconds']) < .2
        assert (folder / 'replay.mp4').stat().st_size < 2_000_000
    assert len(bundle['games']) == 6 and decisions == 25 and tokens == 255957
    total = sum((ROOT / p).stat().st_size for p in manifest['files'])
    assert total < 10_000_000, total
    print(json.dumps(dict(status='passed', files=len(manifest['files']), games=6,
                          decisions=decisions, tokens=tokens, bytes=total), ensure_ascii=False))


if __name__ == '__main__':
    main()
