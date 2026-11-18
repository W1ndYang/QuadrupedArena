"""Reuse a recorded model response only for an identical prompt/API configuration.

This is experimental evidence reuse, not a provider prompt-cache hit. Lookup
never uses a scenario seed or outcome. Misses use the normal live API path.
"""
import copy
import hashlib
import json
from pathlib import Path


API_FIELDS = ('provider', 'base_url', 'model', 'thinking', 'reasoning_effort',
              'max_tokens', 'enable_thinking', 'temperature', 'thinking_budget',
              'response_format')


def api_identity(config):
    return {key: config[key] for key in API_FIELDS if key in config}


def request_digest(identity, messages):
    data = json.dumps(dict(api=identity, messages=messages), sort_keys=True,
                      ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(data.encode()).hexdigest()


def add_recorded_events(entries, events, provenance):
    for event in events:
        if event.get('kind') != 'decision' or event.get('fixture_only') is not False:
            continue
        raw = (event.get('llm_response') or
               event['response'].get('reasoning', {}).get('raw_response'))
        messages = event.get('llm_request')
        if raw is None or not messages:
            continue
        identity = api_identity(event['config'])
        key = request_digest(identity, messages)
        if key in entries:
            if entries[key]['response']['text'] != raw['text']:
                raise ValueError('Conflicting recorded outputs for an identical model request')
            continue
        entries[key] = dict(api=identity, messages=messages, response=raw,
                            provenance=dict(provenance,
                                request_id=event['request']['request_id']))


class ExactResponseCache:
    def __init__(self, path=None):
        self.entries = {}
        if path is not None:
            payload = json.loads(Path(path).read_text())
            if payload.get('schema') != 'llm-exact-response-cache-v1':
                raise ValueError('Unknown recorded-response cache schema')
            self.entries = payload['entries']
            for key, entry in self.entries.items():
                if key != request_digest(entry['api'], entry['messages']):
                    raise ValueError('Recorded-response cache request checksum mismatch')

    def lookup(self, config, messages):
        identity = api_identity(config)
        key = request_digest(identity, messages)
        entry = self.entries.get(key)
        if entry is None:
            return None
        if entry['messages'] != messages or entry['api'] != identity:
            raise ValueError('Recorded-response cache identity mismatch')
        result = copy.deepcopy(entry['response'])
        result['response_source'] = 'recorded_exact_request'
        result['reused_from'] = dict(entry['provenance'], request_sha256=key)
        result['original_model_wall_seconds'] = result.get(
            'original_model_wall_seconds', result.get('wall_seconds'))
        result['wall_seconds'] = 0.
        result['usage_is_reused'] = True
        return result
