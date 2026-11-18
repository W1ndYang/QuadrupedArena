"""Independent HTTP decision service; no Codex process in the runtime path."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import threading
import time

from llm_decision_core import DecisionOrchestrator
from llm_motion_client import APIReasoner, FixtureReasoner, adapter_sources
from llm_map_review import check as check_map
from llm_semantic import COMPACT_SCHEMA_VERSION
from llm_robot_rollout import prewarm


def code_version():
    root=Path(__file__).resolve().parent
    names=('llm_semantic.py','llm_map_review.py','llm_motion_client.py','llm_trajectory.py',
           'llm_formal_optimizer.py','llm_fast_optimizer.py',
           'llm_robot_rollout.py',
           'llm_fast_templates.json','llm_decision_core.py',
           'llm_decision_service.py','llm_option_policy.py',
           'formal_central_screen.py','option_env.py','online_env.py',
           'ppo_observation.py','varied_patrol.py','two_node_geometry.py')
    digest=hashlib.sha256()
    for name in names:
        path=root/name
        if path.exists():digest.update(name.encode()+path.read_bytes())
    for name in ('fixed_goal_probe.py','humanoid_motion.py','trial_motion.py'):
        path=root.parent/'fixed_goal_v041'/'source_snapshot'/'scripts'/name
        digest.update(name.encode()+path.read_bytes())
    for name in ('deepseek_client.py','qwen_client.py','arena_paths.py'):
        path=root.parents[1]/'scripts'/name
        digest.update(name.encode()+path.read_bytes())
    return digest.hexdigest()[:16]


def fixture_response(payload, compact=False):
    """Local engineering smoke only: this is not an LLM research result."""
    request=payload['decision']; region=request['region']
    nodes=payload['map']['nodes']
    edges=[dict(id=edge_id,start=edge_id.split('__',1)[0],
                end=edge_id.split('__',1)[1])
           for edge_id in payload['map']['edge_ids']]
    latest=request['recent_observations'][-1]['xy']
    start=min(nodes,key=lambda n:sum((a-b)**2 for a,b in zip(nodes[n],latest)))
    vy=request['estimated_guard_velocity'][1]
    direction=1 if vy>=0 else -1
    path=[];current=start
    for _ in range(12):
        choices=[e for e in edges if e['start']==current]
        forward=[e for e in choices if
                 direction*(nodes[e['end']][1]-nodes[current][1])>.1]
        if not forward:
            direction*=-1
            forward=[e for e in choices if
                     direction*(nodes[e['end']][1]-nodes[current][1])>.1]
        if not forward:raise ValueError('Fixture map has no connected path')
        selected=max(forward,key=lambda e:(
            direction*(nodes[e['end']][1]-nodes[current][1]),
            -abs(nodes[e['end']][0]-nodes[start][0])))
        path.append(selected)
        current=selected['end']
    response=dict(schema=request['schema'],map_id=request['map_id'],
                track_id=request['track_id'],hypotheses=[
                    dict(id='fixture_path',probability=.8,state='MOVE',
                         next_target_id=path[0]['end'],
                         edge_ids=[e['id'] for e in path],
                         events=[dict(type='MOVE',at_id=e['id']) for e in path],
                         speed_range_mps=[.8,1.5],dwell_range_seconds=[0.,0.],
                         reason='Test fixture; not model inference',
                         uncertainty='True private patrol target unknown'),
                    dict(id='other',probability=.2,state='UNKNOWN',
                         next_target_id=region,edge_ids=[],events=[],
                         speed_range_mps=[.8,1.5],dwell_range_seconds=[0.,4.],
                         reason='Unrepresented routes',uncertainty='Unbounded')])
    if compact:
        response['schema']=COMPACT_SCHEMA_VERSION
        for hypothesis in response['hypotheses']:
            events=hypothesis.pop('events')
            hypothesis['dwell_node_ids']=[event['at_id'] for event in events
                                          if event['type']=='DWELL']
            hypothesis.pop('next_target_id');hypothesis.pop('uncertainty')
    return response


class ServiceState:
    def __init__(self,config,log_dir,fixture=False):
        reasoner=(FixtureReasoner(lambda payload:fixture_response(payload,compact=True))
                  if fixture else APIReasoner(config))
        self.orchestrator=DecisionOrchestrator(reasoner,config)
        prepared_at=time.perf_counter()
        self.startup_precomputation=(dict(prewarm(),call_wall_seconds=time.perf_counter()-prepared_at)
                                    if config.get('robot_rollout_mode')=='stateful_v2' else None)
        manifest=check_map()
        if manifest['map_id']!=self.orchestrator.map['map_id']:
            raise ValueError('Map ID differs from checked geometry')
        self.map_review_status=manifest['review'].get('status')
        if self.map_review_status not in (
                'geometry_verified_manual_semantics_pending','human_approved'):
            raise ValueError('Unknown or invalid map review status')
        self.log_dir=Path(log_dir)
        self.log_dir.mkdir(parents=True,exist_ok=True)
        self.log_path=self.log_dir/'decision_events.jsonl'
        self.lock=threading.RLock()
        self.errors={}
        self.version=code_version()
        self.fixture=fixture
        self.config={k:v for k,v in config.items() if k not in ('api_key','api_key_file')}

    def append(self,event):
        with self.log_path.open('a') as handle:
            handle.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
            handle.flush()

    def decide(self,request):
        key=(request.get('episode_id'),request.get('request_id'))
        with self.lock:
            if key in self.errors:
                previous, response=self.errors[key]
                if previous!=request:
                    raise ValueError('Duplicate failed Request ID has different payload')
                return response
            started=time.perf_counter()
            try:
                response=self.orchestrator.decide(request)
            except Exception as exc:
                response=dict(episode_id=key[0],request_id=key[1],
                              status='error',error_type=type(exc).__name__,
                              error=str(exc))
                context=self.orchestrator.episodes.get(key[0])
                if context is None or key[1] not in context.responses:
                    self.errors[key]=(request,response)
            self.append(dict(kind='decision',utc=datetime.now(timezone.utc).isoformat(),
                             code_version=self.version,config=self.config,
                             map_review_status=self.map_review_status,
                             fixture_only=self.fixture,request=request,
                             response=response,
                             llm_request=(self.orchestrator.episodes.get(key[0])
                                 .llm_messages.get(key[1])
                                 if key[0] in self.orchestrator.episodes else None),
                             llm_response=(self.orchestrator.episodes[key[0]]
                                 .llm_attempts.get(key[1])
                                 if response['status']=='error' and
                                 key[0] in self.orchestrator.episodes else None),
                             service_wall_seconds=time.perf_counter()-started))
            return response

    def finish(self,payload):
        with self.lock:
            if set(payload)!= {'episode_id','summary','observations'}:
                raise ValueError('Invalid finish payload')
            if not isinstance(payload['observations'],list) or len(payload['observations'])>500:
                raise ValueError('Invalid final visual sample batch')
            for row in payload['observations']:
                if set(row)!={'time','xy','observer_xy','observer_yaw',
                             'uncertainty_m','stage','frame_valid'}:
                    raise ValueError('Invalid final visual sample')
                if row['time']>payload['summary']['seconds']+1e-5:
                    raise ValueError('Final visual sample after episode end')
            response=self.orchestrator.finish(payload['episode_id'],payload['summary'],
                                              payload['observations'])
            self.append(dict(kind='finish',utc=datetime.now(timezone.utc).isoformat(),
                             code_version=self.version,
                             map_review_status=self.map_review_status,payload=payload,
                             response=response))
            return response


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*args):
            return

        def _send(self,status,payload):
            body=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers();self.wfile.write(body)

        def do_GET(self):
            if self.path!='/health':
                self._send(404,dict(status='not_found'));return
            self._send(200,dict(status='ready',map_id=state.orchestrator.map['map_id'],
                                rules_version=state.orchestrator.rules['version'],
                                code_version=state.version,fixture_only=state.fixture,
                                api_adapters=adapter_sources(),
                                startup_precomputation=state.startup_precomputation,
                                map_review_status=state.map_review_status))

        def do_POST(self):
            if self.path not in ('/decide','/finish'):
                self._send(404,dict(status='not_found'));return
            try:
                length=int(self.headers.get('Content-Length','-1'))
                if not 0<length<=4_000_000:raise ValueError('Invalid body length')
                payload=json.loads(self.rfile.read(length))
                response=(state.decide(payload) if self.path=='/decide'
                          else state.finish(payload))
                self._send(200 if response.get('status')!='error' else 422,response)
            except Exception as exc:
                self._send(400,dict(status='error',error_type=type(exc).__name__,
                                    error=str(exc)))
    return Handler


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--log-dir',type=Path,required=True)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--fixture-for-tests',action='store_true',
                        help='Engineering smoke only; never count as LLM evaluation')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    state=ServiceState(config,args.log_dir,args.fixture_for_tests)
    server=ThreadingHTTPServer((args.host,args.port),make_handler(state))
    print(json.dumps(dict(status='ready',host=args.host,port=args.port,
                          map_id=state.orchestrator.map['map_id'],
                          fixture_only=args.fixture_for_tests,
                          api_adapters=adapter_sources(),
                          startup_precomputation=state.startup_precomputation,
                          map_review_status=state.map_review_status)),flush=True)
    if state.map_review_status!='human_approved':
        print(json.dumps(dict(status='warning',
            message='Map geometry checked; semantic labels remain unreviewed. '
                    'Treat episode results as provisional. Use llm_map_review.py '
                    '--render-dir MAP_REVIEW_DIR to inspect the map.')),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
