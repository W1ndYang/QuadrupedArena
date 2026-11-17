"""Provider-neutral motion reasoning; credentials never enter decision logs."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import deepseek_client  # noqa: E402
import qwen_client  # noqa: E402

PROMPT_VERSION = 'guard-motion-reasoner-20261117-v1'

SYSTEM_PROMPT = """You infer a moving guard's possible future behavior from public observations.
Return one JSON object only. You never choose the robot action and never output
continuous per-frame coordinates. Use only supplied map IDs and legal observation
history. The four private patrol waypoints and their order are unknown; propose
mutually exclusive, plausible routes through map geometry, not the true hidden
route. Include an UNKNOWN hypothesis whenever your routes omit material
possibilities. Probabilities are numerical predictions summing to 1.

JSON fields: schema='guard-motion-hypotheses-v1', map_id, track_id,
hypotheses=[{id,probability,state,next_target_id,edge_ids,events,
speed_range_mps,dwell_range_seconds,reason,uncertainty}].
state is MOVE, TURN, DWELL, REVERSE or UNKNOWN. Each event is
{type: MOVE|TURN|DWELL|REVERSE, at_id: existing Node or Edge ID}.
edge_ids form a connected directed path in the current guard region.
The edge sequence and explicit DWELL events must cover the supplied prediction
horizon. Do not end a path early and assume the guard stops. The first edge
must start within 2.25 m of the latest guard sighting. Reverse travel uses
the separately listed reverse directed edge ID.
For UNKNOWN use edge_ids=[] and events=[]. Supply 1 to 6 hypotheses.
Do not invent map IDs or claim that a spatial edge is a known patrol route.
"""


def prompt_digest():
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16]


def messages_for(request, semantic_map, rules):
    region = request['region']
    node_ids = semantic_map['regions'][region]['node_ids']
    edge_rows = [dict(id=e['id'], start=e['start'], end=e['end'],
                      length_m=e['length_m'])
                 for e in semantic_map['edges'].values()
                 if 'guard' in e['traversable_by'] and e['start'] in node_ids]
    public_map = dict(map_id=semantic_map['map_id'], region=semantic_map['regions'][region],
                      nodes={n: semantic_map['nodes'][n]['xy'] for n in node_ids},
                      edges=edge_rows,
                      obstacles=semantic_map['obstacles'])
    payload = dict(map=public_map, rules=rules, decision=request)
    return [dict(role='system', content=SYSTEM_PROMPT),
            dict(role='user', content=json.dumps(payload, ensure_ascii=False,
                                                allow_nan=False))]


def _key(config):
    env_name = config.get('api_key_env', 'DEEPSEEK_API_KEY')
    value = os.environ.get(env_name)
    if value:
        if any(c.isspace() for c in value):
            raise ValueError(f'{env_name} contains whitespace')
        return value
    file_name = config.get('api_key_file')
    if file_name:
        return qwen_client.load_key(Path(file_name).expanduser())
    raise FileNotFoundError(f'Set {env_name} or api_key_file in the private config')


class APIReasoner:
    def __init__(self, config):
        self.config = dict(config)
        self.provider = config['provider']
        if self.provider not in ('deepseek', 'qwen'):
            raise ValueError('Unsupported provider')

    def infer(self, messages):
        key = _key(self.config)
        if self.provider == 'deepseek':
            result = deepseek_client.call_api(self.config, key, messages)
        else:
            result = qwen_client.call_api(self.config, key, messages)
        if result.get('tool_calls'):
            raise ValueError('Motion reasoner must not call tools')
        text = result.get('content', '')
        if not isinstance(text, str) or len(text) > 100_000:
            raise ValueError('Empty or oversized LLM response')
        return dict(text=text, usage=result.get('usage'),
                    model=result.get('response_model'),
                    response_id=result.get('response_id'),
                    wall_seconds=result.get('api_wall_seconds'),
                    raw_metadata={k: result.get(k) for k in
                                  ('finish_reason', 'api_started_utc',
                                   'api_finished_utc', 'reasoning_characters_received')})


class FixtureReasoner:
    """Explicit test-only provider; production CLI never selects it."""
    def __init__(self, callback):
        self.callback = callback

    def infer(self, messages):
        response = self.callback(json.loads(messages[-1]['content']))
        return dict(text=json.dumps(response), usage=None, model='fixture-only',
                    response_id=None, wall_seconds=0., raw_metadata={})
