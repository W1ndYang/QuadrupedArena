"""Provider-neutral motion reasoning; credentials never enter decision logs."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

from llm_response_cache import ExactResponseCache

ROOT = Path(__file__).resolve().parents[2]
# The frozen simulator imports modules with the same names into sys.modules.
# A separate package binds this runtime to the current adapters and their
# relative dependencies without replacing the frozen experiment's imports.
_PACKAGE='_arena_llm_api_runtime'
if _PACKAGE not in sys.modules:
    package=types.ModuleType(_PACKAGE)
    package.__path__=[str(ROOT/'scripts')]
    sys.modules[_PACKAGE]=package
deepseek_client=importlib.import_module(f'{_PACKAGE}.deepseek_client')
qwen_client=importlib.import_module(f'{_PACKAGE}.qwen_client')


def adapter_sources():
    return {name:dict(path=str(Path(module.__file__).resolve().relative_to(ROOT)),
                      sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
            for name,module in (('deepseek',deepseek_client),('qwen',qwen_client))}


PROMPT_VERSION = 'guard-motion-reasoner-20261118-v7'

SYSTEM_PROMPT = """You infer a moving guard's possible future behavior from public observations.
Return one JSON object only. You never choose the robot action and never output
continuous per-frame coordinates. Use only supplied map IDs and legal observation
history. The four private patrol waypoints and their order are unknown; propose
mutually exclusive, plausible routes through map geometry, not the true hidden
route. Include an UNKNOWN hypothesis whenever your routes omit material
possibilities. Probabilities are numerical predictions summing to 1.

JSON fields: schema='guard-motion-paths-v2', map_id, track_id,
hypotheses=[{id,probability,state,edge_ids,dwell_node_ids,
speed_range_mps,dwell_range_seconds,reason}].
state is MOVE, TURN, DWELL, REVERSE or UNKNOWN.
Time contract: paths, state and remaining dwell refer to latest_measurement_at,
not to API return time. The runtime advances through measurement_age_seconds
before using a prediction at game_time. Cover path_coverage_seconds from that
measurement, including the blind interval and the requested future horizon.
edge_ids form a connected directed path in the current guard region.
An edge ID has the exact form start_node_id__end_node_id; copy it from
map.edge_ids rather than constructing a new ID.
The edge sequence and dwell at dwell_node_ids must cover the supplied prediction
horizon. Do not end a path early and assume the guard stops. The first edge
must start within 2.25 m of the latest guard sighting. Reverse travel uses
the separately listed reverse directed edge ID.
Use dwell_node_ids only for nodes on that path. For UNKNOWN use edge_ids=[]
and dwell_node_ids=[]. Prefer 2 or 3 substantive alternatives plus UNKNOWN;
use at most 6 hypotheses if needed. Keep each reason to at most 12 words.
The runtime derives MOVE events and next target from edge_ids; omit events,
next_target_id and uncertainty. Avoid repeating the same path in prose.
Do not invent map IDs or claim that a spatial edge is a known patrol route.
"""


def prompt_digest():
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16]


def messages_for(request, semantic_map, rules):
    region = request['region']
    node_ids = semantic_map['regions'][region]['node_ids']
    edge_ids = [e['id'] for e in semantic_map['edges'].values()
                if 'guard' in e['traversable_by'] and e['start'] in node_ids]
    public_map = dict(map_id=semantic_map['map_id'], region=semantic_map['regions'][region],
                      nodes={n: semantic_map['nodes'][n]['xy'] for n in node_ids},
                      edge_ids=edge_ids,
                      obstacles=semantic_map['obstacles'])
    payload = dict(map=public_map, rules=rules,
                   response_constraints=dict(
                       output_schema='guard-motion-paths-v2',
                       allowed_dwell_node_ids=node_ids,
                       region_id_for_unknown=region),
                   decision=request)
    return [dict(role='system', content=SYSTEM_PROMPT),
            dict(role='user', content=json.dumps(payload, ensure_ascii=False,
                                                allow_nan=False,
                                                separators=(',',':')))]


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
        self.recorded = ExactResponseCache(config.get('recorded_response_cache_file'))
        if self.provider not in ('deepseek', 'qwen'):
            raise ValueError('Unsupported provider')
        if self.provider=='deepseek':
            deepseek_client.payload(self.config,[])
        else:
            qwen_client.endpoint(self.config)

    def infer(self, messages):
        recorded = self.recorded.lookup(self.config, messages)
        if recorded is not None:
            return recorded
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
        return dict(text=text, usage=result.get('usage'),response_source='live_api',
                    model=result.get('response_model'),
                    response_id=result.get('response_id'),
                    wall_seconds=result.get('api_wall_seconds'),
                    prompt_utf8_bytes=sum(len(row['content'].encode('utf-8'))
                                          for row in messages),
                    raw_metadata={k: result.get(k) for k in
                                  ('finish_reason', 'api_started_utc',
                                   'api_finished_utc', 'reasoning_characters_received',
                                   'time_to_first_output_seconds',
                                   'time_to_first_visible_content_seconds')})


class FixtureReasoner:
    """Explicit test-only provider; production CLI never selects it."""
    def __init__(self, callback):
        self.callback = callback

    def infer(self, messages):
        response = self.callback(json.loads(messages[-1]['content']))
        return dict(text=json.dumps(response), usage=None, model='fixture-only',
                    response_id=None, wall_seconds=0., raw_metadata={})
