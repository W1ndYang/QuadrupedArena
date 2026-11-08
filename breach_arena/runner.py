"""Build controllers and run rounds, two-round series and the play/counter-play matrix."""
import json
import time
from pathlib import Path

from .agent import QuadAgent
from .config import GameMap, load_rules
from .engine import Match
from .humans import make_bot
from .planner_rule import ReadAndCounter, ScriptedDecider
from .replay import build_replay, save_replay


def make_decider(quad, game_map, rules, llm_config=None, key_file=None, timing='virtual', hints='routes',
                 strategy=True, record_dir=None, fake_delay=0.0):
    """quad: 'rule' | 'llm' | 'fake-llm' | {'script': [...]} | path to a JSON script."""
    if isinstance(quad, dict):
        return ScriptedDecider(quad['script']), 'scripted'
    if quad == 'rule':
        return ReadAndCounter(), 'rule:read-and-counter'
    if quad in ('llm', 'fake-llm'):
        from .llm.broker import LLMDecider
        if quad == 'fake-llm':
            from .llm.client import FakeClient
            from .llm.fake import RulePolicyModel
            client = FakeClient(RulePolicyModel(game_map, rules), delay_s=fake_delay)
        else:
            from .llm.client import ChatClient, load_config, resolve_key
            cfg = load_config(llm_config)
            client = ChatClient(cfg, resolve_key(cfg, key_file))
        dec = LLMDecider(client, timing=timing, hints=hints, strategy=strategy, record_dir=record_dir)
        return dec, f"llm:{client.public_config().get('model')}"
    p = Path(quad)
    if p.suffix == '.json' and p.exists():
        return ScriptedDecider(json.loads(p.read_text(encoding='utf-8'))), f'scripted:{p.name}'
    raise ValueError(f'unknown quadruped controller {quad!r}')


def run_round(game_map, rules, quad, human, out_dir=None, round_index=1, history=None, pace=None, on_tick=None,
              **decider_kw):
    """One match. `human` is a bot spec or a controller object. Returns (replay, viewer_path)."""
    record = Path(out_dir) / f'round{round_index}_api' if out_dir else None
    decider, quad_label = make_decider(quad, game_map, rules, record_dir=record, **decider_kw)
    agent = QuadAgent(decider, history=history)
    human_ctrl = make_bot(human) if isinstance(human, str) else human
    human_label = f'bot:{human}' if isinstance(human, str) else getattr(human_ctrl, 'label', 'player')
    match = Match(game_map, rules, agent, human_ctrl)
    started = time.perf_counter()
    if pace:                                    # realtime: keep match time <= wall time * pace
        while True:
            if not match.step():
                break
            if on_tick:
                on_tick(match)
            lag = match.t / pace - (time.perf_counter() - started)
            if lag > 0:
                time.sleep(lag)
    else:
        match.run(on_tick=on_tick)
    timing = decider_kw.get('timing', 'virtual') if quad in ('llm', 'fake-llm') else 'sync'
    replay = build_replay(match, agent, dict(quadruped=quad_label, human=human_label), timing, round_index,
                          getattr(decider, 'calls', None))
    replay['wall_seconds'] = round(time.perf_counter() - started, 2)
    replay['next_round_history'] = (history or []) + [dict(round=round_index, **agent.round_summary(match.result))]
    viewer = save_replay(replay, out_dir) if out_dir else None
    return replay, viewer


def run_series(game_map, rules, quad, humans, out_dir=None, **kw):
    """Several rounds; the quadruped keeps a legal summary of earlier rounds (no hidden truth)."""
    history, replays = [], []
    for i, human in enumerate(humans, 1):
        rep, _ = run_round(game_map, rules, quad, human, out_dir, i, history, **kw)
        history = rep['next_round_history']
        replays.append(rep)
    return replays


def run_matrix(map_id, rules=None, out_dir=None, only=None):
    game_map = GameMap.load(map_id)
    rules = rules or load_rules()
    rows = []
    for case in game_map.data.get('cycle_checks', []):
        if only and case['id'] not in only:
            continue
        sub = Path(out_dir) / case['id'] if out_dir else None
        rep, _ = run_round(game_map, rules, case['quad'], case['human'], sub)
        res = rep['summary']['result']
        rows.append(dict(id=case['id'], human=case['human'],
                         quad='rule' if case['quad'] == 'rule' else 'script', expect=case['expect'],
                         winner=res['winner'], reason=res['reason'], t=res['t'], ok=res['winner'] == case['expect'],
                         why=case.get('why', '')))
    return rows
