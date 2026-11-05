"""Small audited DashScope HTTP/SSE adapter. No SDK or automatic retries.

Credentials live outside the project and experiment trees. Never print raw
HTTP exceptions, headers, response bodies, or reasoning_content.
"""
import getpass
import datetime
import json
import os
from pathlib import Path
import socket
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from arena_paths import credential_path
KEY_PATH=credential_path('dashscope.key')


class APIError(RuntimeError):pass


def endpoint(config):
    base=config['base_url'].rstrip('/');parts=urllib.parse.urlsplit(base)
    host=parts.hostname or ''
    if parts.scheme!='https' or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Only HTTPS endpoints without URL credentials/query are allowed')
    if not (host=='dashscope.aliyuncs.com' or host.endswith('.cn-beijing.maas.aliyuncs.com')):
        raise ValueError('This Beijing profile only sends credentials to official Beijing DashScope hosts')
    if parts.port not in (None,443):raise ValueError('Unexpected API port')
    if parts.path!='/compatible-mode/v1':raise ValueError('Expected /compatible-mode/v1 Base URL')
    return base+'/chat/completions'


def save_key(key,path=KEY_PATH,replace=False):
    if not isinstance(key,str) or len(key.strip())<16 or any(c.isspace() for c in key):
        raise ValueError('API Key looks empty or contains whitespace; nothing saved')
    path=Path(path)
    if path.is_symlink() or path.parent.is_symlink():raise ValueError('Refusing symlink credential target')
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(path.parent,0o700)
    flags=os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW|(os.O_TRUNC if replace else os.O_EXCL)
    fd=os.open(path,flags,0o600)
    with os.fdopen(fd,'w') as file:
        os.fchmod(file.fileno(),0o600);file.write(key.strip()+'\n')


def load_key(path=KEY_PATH):
    path=Path(path)
    if path.is_symlink():raise ValueError('Refusing symlink credential file')
    if stat.S_IMODE(path.stat().st_mode)&0o077:raise ValueError('Credential file must have owner-only permissions (0600)')
    key=path.read_text().strip()
    if len(key)<16 or any(c.isspace() for c in key):raise ValueError('Invalid stored API Key')
    return key


def setup_key(replace=False):
    if not sys.stdin.isatty():raise ValueError('Run setup in your local terminal: hidden input requires a TTY')
    if KEY_PATH.exists() and not replace:raise ValueError('Key already configured; use setup --replace only if you intend to replace it')
    key=getpass.getpass('请粘贴百炼北京 API Key（不显示输入，粘贴后回车）：')
    save_key(key,replace=replace)
    print('API Key 已保存到本机私密文件，权限 0600；未发送任何 API 请求。')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        raise APIError('API redirect rejected; credentials were not forwarded')


def parse_stream(lines,started,clock=time.perf_counter):
    content=[];reasoning_chars=0;first_any=None;first_content=None;usage=None;model=None;request_id=None;finish=None;calls={};done=False
    for raw in lines:
        if not raw.startswith(b'data:'):continue
        body=raw[5:].strip()
        if body==b'[DONE]':done=True;break
        try:chunk=json.loads(body)
        except (ValueError,UnicodeDecodeError):raise APIError('Malformed SSE JSON from API') from None
        if chunk.get('error'):raise APIError('API returned an error event; body suppressed')
        if chunk.get('usage') is not None:usage=chunk['usage']
        model=chunk.get('model') or model;request_id=chunk.get('id') or request_id
        for choice in chunk.get('choices',[]):
            if choice.get('index',0)!=0:continue
            delta=choice.get('delta') or {}
            if delta.get('reasoning_content') or delta.get('content') or delta.get('tool_calls'):
                if first_any is None:first_any=clock()-started
            if delta.get('reasoning_content'):reasoning_chars+=len(delta['reasoning_content'])
            if isinstance(delta.get('content'),str) and delta['content']:
                if first_content is None:first_content=clock()-started
                content.append(delta['content'])
            for call in delta.get('tool_calls',[]):
                slot=calls.setdefault(call['index'],dict(id='',type='function',function=dict(name='',arguments='')))
                if call.get('id'):slot['id']+=call['id']
                for field in ('name','arguments'):
                    if call.get('function',{}).get(field):slot['function'][field]+=call['function'][field]
            if choice.get('finish_reason') is not None:finish=choice['finish_reason']
    if not done or finish is None:raise APIError('Incomplete API stream; simulation remains paused')
    return dict(content=''.join(content),tool_calls=[calls[i] for i in sorted(calls)],usage=usage,
        response_model=model,response_id=request_id,finish_reason=finish,
        api_wall_seconds=clock()-started,time_to_first_output_seconds=first_any,
        time_to_first_visible_content_seconds=first_content,reasoning_characters_received=reasoning_chars,
        reasoning_text_saved=False)


def call_api(config,key,messages,tools=None):
    payload=dict(model=config['model'],messages=messages,stream=True,stream_options=dict(include_usage=True),
        enable_thinking=config['enable_thinking'],max_tokens=config['max_tokens'],temperature=config['temperature'])
    if config['enable_thinking'] and config.get('thinking_budget'):payload['thinking_budget']=config['thinking_budget']
    if config.get('response_format'):payload['response_format']=config['response_format']
    if tools:
        choice=config.get('tool_choice','auto')
        if choice not in ('auto','none'):raise ValueError('Unsupported tool choice')
        payload['tools']=tools;payload['tool_choice']=choice
    request=urllib.request.Request(endpoint(config),data=json.dumps(payload,ensure_ascii=False).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'text/event-stream'},method='POST')
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat();started=time.perf_counter()
    try:
        with urllib.request.build_opener(NoRedirect()).open(request,timeout=config['timeout_seconds']) as response:
            def lines():
                for line in response:
                    if config.get('wall_timeout_seconds') and time.perf_counter()-started>config['wall_timeout_seconds']:
                        raise APIError('API wall-time budget exceeded; no retry; request may have been billed')
                    yield line
            result=parse_stream(lines(),started)
            result.update(api_started_utc=started_utc,api_finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except urllib.error.HTTPError as error:
        raise APIError(f'API HTTP {error.code}; check region/key/model/quota or request parameters. Response body suppressed.') from None
    except (urllib.error.URLError,TimeoutError,socket.timeout,OSError):
        raise APIError('API network/timeout failure; no automatic retry; request may have been billed') from None
    # Defensive redaction in case a service unexpectedly echoes a credential.
    return json.loads(json.dumps(result,ensure_ascii=False).replace(key,'[REDACTED]'))


def usage_total(records):
    fields=('prompt_tokens','completion_tokens','total_tokens')
    usages=[r.get('usage') for r in records]
    result={key:sum(u[key] for u in usages) if usages and all(isinstance(u,dict) and type(u.get(key)) is int for u in usages) else None for key in fields}
    details=[u.get('completion_tokens_details',{}) if isinstance(u,dict) else {} for u in usages]
    result['reasoning_tokens']=sum(d['reasoning_tokens'] for d in details) if details and all(isinstance(d,dict) and type(d.get('reasoning_tokens')) is int for d in details) else None
    result['calls']=len(records);result['usage_complete']=all(result[k] is not None for k in fields)
    return result
