"""Rules and map loading. Plain JSON in, attribute-style rules out."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from . import MAP_SCHEMA, RULES_VERSION

PACKAGE = Path(__file__).resolve().parent
MAP_DIR = PACKAGE / 'maps'


def _ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in value.items()})
    return value


def load_rules(path=None, overrides=None):
    raw = json.loads(Path(path or PACKAGE / 'rules.json').read_text(encoding='utf-8'))
    if raw.get('version') != RULES_VERSION:
        raise ValueError(f"rules version {raw.get('version')!r} is not {RULES_VERSION}")
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split('.')
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f'unknown rule {dotted}')
        node[leaf] = value
    rules = _ns(raw)
    rules.raw = raw
    return rules


def available_maps():
    return sorted(p.stem for p in MAP_DIR.glob('BA-*.json'))


class GameMap:
    """Static map: walls, goal, spawns and named points. Public to both sides."""

    def __init__(self, data):
        if data.get('schema') != MAP_SCHEMA:
            raise ValueError(f"map schema {data.get('schema')!r} is not {MAP_SCHEMA}")
        self.data = data
        self.id, self.title = data['id'], data['title']
        self.size = tuple(data['arena']['size'])
        self.covers = data['arena']['covers']
        self.goal = tuple(data['goal']['center'])
        self.goal_radius = data['goal']['radius']
        self.spawns = data['spawns']
        self.points = {k: tuple(v['xy']) for k, v in data['points'].items()}
        self.kinds = {k: v['kind'] for k, v in data['points'].items()}
        self.labels = {k: v.get('label', k) for k, v in data['points'].items()}
        self.plays = data.get('plays', {})
        self.bots = data.get('human_bots', {})

    @classmethod
    def load(cls, name_or_path):
        p = Path(name_or_path)
        if not p.suffix:
            p = MAP_DIR / f'{name_or_path}.json'
        return cls(json.loads(p.read_text(encoding='utf-8')))

    def resolve(self, spec):
        """A named point or an explicit [x, y] pair -> (x, y)."""
        if isinstance(spec, str):
            if spec == 'GOAL':
                return self.goal
            if spec not in self.points:
                raise KeyError(f'unknown point {spec!r} on {self.id}')
            return self.points[spec]
        if isinstance(spec, (list, tuple)) and len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            return (float(spec[0]), float(spec[1]))
        raise ValueError(f'bad point spec {spec!r}')

    def names(self, kind):
        return [k for k, v in self.kinds.items() if v == kind]

    def public_dict(self):
        """What both players may know before the match."""
        return copy.deepcopy(self.data)
