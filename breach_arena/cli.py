"""Command line: python -m breach_arena <command> ...

  play     one round, bot human vs quadruped controller; saves replay + HTML viewer
  series   several rounds (e.g. --humans hold,hunt); the quadruped keeps a legal round summary
  matrix   the map's play / counter-play checks (deterministic; used as regression)
  checks   static map design checks (sight lines, smoke matrix, lure timing) + overview figure
  live     you play the human (pygame window), quadruped by rule or LLM, real-time, no pausing
  prompt   print the LLM system prompt for a map
  viewer   rebuild the HTML viewer from a replay file
"""
import argparse
import json
import sys

from .config import GameMap, available_maps, load_rules
from .replay import new_run_dir


def _quad_args(p):
    p.add_argument('--quad', default='rule', help='rule | llm | fake-llm | path/to/script.json')
    p.add_argument('--llm-config', help='provider JSON, e.g. breach_arena/llm/presets/deepseek.json')
    p.add_argument('--key-file', help='read the API key from this file instead of the env var named in the config')
    p.add_argument('--timing', default=None,
                   help='LLM timing: virtual (default headless) | realtime | pause | fixed:SECONDS')
    p.add_argument('--hints', default='routes', choices=['none', 'routes', 'full'],
                   help='map play hints in the prompt (full adds smoke landing hints)')
    p.add_argument('--no-strategy', action='store_true', help='omit the strategy paragraph from the prompt')
    p.add_argument('--fake-delay', type=float, default=0.0, help='fake-llm wall latency per call (s)')


def _decider_kw(a, default_timing):
    return dict(llm_config=a.llm_config, key_file=a.key_file, timing=a.timing or default_timing,
                hints=a.hints, strategy=not a.no_strategy, fake_delay=a.fake_delay)


def _check_llm(a):
    if a.quad == 'llm' and not a.llm_config:
        sys.exit('--quad llm needs --llm-config (see breach_arena/llm/presets/)')


def _print_round(rep, viewer):
    s = rep['summary']
    res = s['result']
    print(f"round {rep['round']}: {res['winner']} ({res['reason']}) at {res['t']} s   "
          f"quad={rep['players']['quadruped']} human={rep['players']['human']} smoke_used={s['smoke_used']}")
    for e in rep['quadruped']['log']:
        if e['type'] == 'response':
            plan = ' / '.join(st['skill'] + ('→' + str(st['params'].get('to') or st['params'].get('throw_from') or ''))
                              for st in (e['plan'] or []))
            lat = '' if e.get('latency_s') is None else f" latency {e['latency_s']} s"
            print(f"   [{e['t']:6.2f}s{lat}] {plan or 'ERROR ' + str(e.get('error'))} | "
                  f"{(e.get('assessment') or {}).get('decision_summary', '')}")
    if viewer:
        print(f'   viewer: {viewer}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m breach_arena', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    maps = available_maps()

    p = sub.add_parser('play', help='one round with a bot human')
    p.add_argument('--map', default='BA-1', choices=maps)
    p.add_argument('--human', default='hold', help='hold | standoff:POINT | hunt | hunt:SECONDS')
    p.add_argument('--out', help='output folder (default .arena-workspace/breach_arena/runs/...)')
    _quad_args(p)

    p = sub.add_parser('series', help='several rounds, quadruped remembers earlier rounds')
    p.add_argument('--map', default='BA-1', choices=maps)
    p.add_argument('--humans', default='hold,hunt', help='comma separated bot specs, one per round')
    p.add_argument('--out')
    _quad_args(p)

    p = sub.add_parser('matrix', help="run the map's play/counter-play checks")
    p.add_argument('--map', default='BA-1', choices=maps)
    p.add_argument('--out', help='also save every replay here')
    p.add_argument('--json', action='store_true')

    p = sub.add_parser('checks', help='static design checks + overview figure')
    p.add_argument('--map', nargs='*', default=None)
    p.add_argument('--out', help='folder for checks JSON and overview.png (default: run folder)')

    p = sub.add_parser('live', help='play the human yourself (pygame)')
    p.add_argument('--map', default='BA-1', choices=maps)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--out')
    _quad_args(p)

    p = sub.add_parser('prompt', help='print the LLM system prompt')
    p.add_argument('--map', default='BA-1', choices=maps)
    p.add_argument('--hints', default='routes', choices=['none', 'routes', 'full'])
    p.add_argument('--no-strategy', action='store_true')

    p = sub.add_parser('viewer', help='rebuild HTML viewer from a replay')
    p.add_argument('replay')
    p.add_argument('--out')

    a = ap.parse_args(argv)
    rules = load_rules()

    if a.cmd == 'play':
        from .runner import run_round
        _check_llm(a)
        out = a.out or new_run_dir(f'{a.map}_{a.human.replace(":", "-")}_{a.quad if a.quad in ("rule", "llm", "fake-llm") else "script"}')
        rep, viewer = run_round(GameMap.load(a.map), rules, a.quad, a.human, out, **_decider_kw(a, 'virtual'))
        _print_round(rep, viewer)
        return 0
    if a.cmd == 'series':
        from .runner import run_series
        _check_llm(a)
        humans = [h.strip() for h in a.humans.split(',') if h.strip()]
        out = a.out or new_run_dir(f'{a.map}_series_{"-".join(h.replace(":", "") for h in humans)}')
        reps = run_series(GameMap.load(a.map), rules, a.quad, humans, out, **_decider_kw(a, 'virtual'))
        for i, rep in enumerate(reps, 1):
            _print_round(rep, f'{out}/round{i}.viewer.html')
        return 0
    if a.cmd == 'matrix':
        from .runner import run_matrix
        rows = run_matrix(a.map, rules, a.out)
        if a.json:
            print(json.dumps(rows, ensure_ascii=False, indent=1))
        else:
            for r in rows:
                print(f"{'OK ' if r['ok'] else 'XX '} {r['id']:<32} human={r['human']:<15} quad={r['quad']:<6} "
                      f"expect={r['expect']:<9} got={r['winner']:<9} {r['reason']:<7} {r['t']:>6} s  {r['why']}")
            bad = [r for r in rows if not r['ok']]
            print(f'{len(rows) - len(bad)}/{len(rows)} as designed')
        return 0 if all(r['ok'] for r in rows) else 1
    if a.cmd == 'checks':
        from .design_checks import run_checks
        out = a.out or new_run_dir('design_checks')
        summary = run_checks(a.map or maps, out, rules)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        print(f'written to {out}')
        return 0
    if a.cmd == 'live':
        _check_llm(a)
        try:
            import pygame  # noqa: F401
        except ImportError:
            sys.exit('live play needs pygame: python -m pip install -r breach_arena/requirements-live.txt')
        from .live import play_live
        out = a.out or new_run_dir(f'{a.map}_live')
        results = play_live(a.map, a.quad, a.rounds, out, **_decider_kw(a, 'realtime'))
        print(json.dumps(results, ensure_ascii=False, indent=1))
        return 0
    if a.cmd == 'prompt':
        from .llm.protocol import system_prompt
        print(system_prompt(GameMap.load(a.map), rules, a.hints, strategy=not a.no_strategy))
        return 0
    if a.cmd == 'viewer':
        from .replay import load_replay, write_viewer
        rep = load_replay(a.replay)
        out = a.out or str(a.replay).replace('.replay.json.gz', '.viewer.html').replace('.json', '.viewer.html')
        print(write_viewer(rep, out))
        return 0
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
