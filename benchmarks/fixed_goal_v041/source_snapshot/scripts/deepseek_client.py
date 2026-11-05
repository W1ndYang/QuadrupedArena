"""Official DeepSeek only. No retries, redirects, raw errors or saved reasoning."""
import json
import datetime
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request
from qwen_client import APIError,NoRedirect,parse_stream,save_key,load_key as read_private_key,usage_total

from arena_paths import credential_path
KEY_PATH=credential_path('deepseek.key')


def load_key():return read_private_key(KEY_PATH)


def endpoint(config):
    if config['base_url'] not in ('https://api.deepseek.com','https://api.deepseek.com/'):
        raise ValueError('Only the official https://api.deepseek.com endpoint is allowed')
    return 'https://api.deepseek.com/chat/completions'


def payload(config,messages):
    endpoint(config)
    if config['model']!='deepseek-flash' or config['thinking']!={'type':'enabled'} or config['reasoning_effort'] not in ('low','high','max'):
        raise ValueError('This experiment requires deepseek-flash with thinking enabled and low/high/max effort')
    result=dict(model=config['model'],messages=messages,thinking=config['thinking'],reasoning_effort=config['reasoning_effort'],
        stream=True,stream_options=dict(include_usage=True),response_format=dict(type='json_object'))
    limit=config.get('max_tokens')
    if limit is not None:
        if type(limit) is not int or not 1<=limit<=393216:raise ValueError('max_tokens must be 1..393216, or omitted for provider defaults')
        result['max_tokens']=limit
    return result


def call_api(config,key,messages):
    body=json.dumps(payload(config,messages),ensure_ascii=False,allow_nan=False).encode()
    if len(body)>48*1024*1024:raise ValueError('Request exceeds official 48 MiB limit')
    request=urllib.request.Request(endpoint(config),data=body,method='POST',
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'text/event-stream'})
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat();started=time.perf_counter()
    wall_timeout=config.get('wall_timeout_seconds',config['timeout_seconds'])
    try:
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=config['timeout_seconds']) as response:
            def bounded_lines():
                for line in response:
                    if wall_timeout is not None and time.perf_counter()-started>wall_timeout:
                        raise APIError('API response exceeded wall-time budget; no retry; request may have been billed')
                    yield line
            result=parse_stream(bounded_lines(),started)
            result.update(api_started_utc=started_utc,api_finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                api_wall_seconds=time.perf_counter()-started)
    except urllib.error.HTTPError as error:
        raise APIError(f'DeepSeek HTTP {error.code}; check key/model/balance/parameters. Response body suppressed.') from None
    except (urllib.error.URLError,TimeoutError,socket.timeout,OSError):
        raise APIError('DeepSeek network/timeout failure; no retry; request may have been billed') from None
    return json.loads(json.dumps(result,ensure_ascii=False).replace(key,'[REDACTED]'))
