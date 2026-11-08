"""Minimal OpenAI-compatible chat client (stdlib only). No retries, no redirects; key never logged.

Config JSON (see presets/): base_url, model, api_key_env, timeout_s, and optional
temperature / max_tokens / extra (provider-specific body fields, e.g. reasoning settings).
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


class APIError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    for k in ('base_url', 'model', 'api_key_env', 'timeout_s'):
        if k not in cfg:
            raise ValueError(f'LLM config {path} missing {k}')
    if not cfg['base_url'].startswith('https://'):
        raise ValueError('base_url must be https')
    return cfg


def resolve_key(cfg, key_file=None):
    if key_file:
        key = Path(key_file).read_text(encoding='utf-8').strip()
    else:
        key = os.environ.get(cfg['api_key_env'], '').strip()
    if not key:
        raise APIError(f"no API key: set ${cfg['api_key_env']} or pass --key-file")
    return key


class ChatClient:
    def __init__(self, cfg, key):
        self.cfg, self._key = cfg, key

    def public_config(self):
        return {k: v for k, v in self.cfg.items() if 'key' not in k.lower() or k == 'api_key_env'}

    def complete(self, messages):
        cfg = self.cfg
        body = dict(model=cfg['model'], messages=messages, response_format={'type': 'json_object'}, stream=False)
        for k in ('temperature', 'max_tokens'):
            if cfg.get(k) is not None:
                body[k] = cfg[k]
        body.update(cfg.get('extra', {}))
        req = urllib.request.Request(cfg['base_url'].rstrip('/') + '/chat/completions', method='POST',
                                     data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={'Authorization': 'Bearer ' + self._key, 'Content-Type': 'application/json'})
        started = time.perf_counter()
        try:
            with urllib.request.build_opener(_NoRedirect()).open(req, timeout=cfg['timeout_s']) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            raise APIError(f'HTTP {e.code} from provider (body suppressed); no retry') from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise APIError('network error or timeout; no retry') from None
        wall = time.perf_counter() - started
        choice = (data.get('choices') or [{}])[0].get('message', {})
        text = choice.get('content') or ''
        out = dict(text=text, usage=data.get('usage'), model=data.get('model'), wall_s=round(wall, 3),
                   reasoning_chars=len(choice.get('reasoning_content') or ''))
        return json.loads(json.dumps(out, ensure_ascii=False).replace(self._key, '[REDACTED]'))


class FakeClient:
    """Offline stand-in: answers with a callback(packet_json_dict) -> response dict, after `delay_s` of wall time."""

    def __init__(self, respond, delay_s=0.0):
        self.respond, self.delay_s = respond, delay_s
        self.cfg = dict(model='fake', base_url='offline')

    def public_config(self):
        return dict(self.cfg)

    def complete(self, messages):
        started = time.perf_counter()
        if self.delay_s:
            time.sleep(self.delay_s)
        user = json.loads(messages[-1]['content'])
        text = json.dumps(self.respond(user), ensure_ascii=False)
        return dict(text=text, usage=None, model='fake', wall_s=round(time.perf_counter() - started, 3), reasoning_chars=0)
