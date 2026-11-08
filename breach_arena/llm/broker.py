"""LLM decider for QuadAgent. The match keeps running while a request is in flight.

timing modes
  realtime   requests run in a background thread; the runner paces the match to the wall clock
             (live play with a human). A plan is applied on the tick it arrives.
  virtual    same background thread, match not paced: while a request is in flight the match may
             not run ahead of the wall time since the request was sent, so the plan lands exactly
             `measured latency` of match time after the request. Idle time is compressed. Headless,
             faithful to real latency.
  fixed:S    deliver S seconds of match time after the request regardless of wall time (tests).
  pause      blocking call; the match waits (comparison with earlier pause-per-decision studies).
"""
import json
import threading
import time
from pathlib import Path

from .protocol import PlanError, compact_packet, parse_response, system_prompt


class LLMDecider:
    def __init__(self, client, timing='virtual', hints='routes', strategy=True, language='zh', record_dir=None):
        self.client, self.timing = client, timing
        self.hints, self.strategy, self.language = hints, strategy, language
        self.record_dir = Path(record_dir) if record_dir else None
        self.inflight = None
        self.done = []
        self.calls = []
        self.lock = threading.Lock()
        if timing.startswith('fixed:'):
            self.fixed = float(timing.split(':', 1)[1])
        elif timing in ('realtime', 'virtual', 'pause'):
            self.fixed = None
        else:
            raise ValueError(f'unknown timing {timing!r}')

    def attach(self, game_map, rules):
        self.map, self.rules = game_map, rules
        self.system = system_prompt(game_map, rules, self.hints, self.language, self.strategy)
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / 'system_prompt.txt').write_text(self.system, encoding='utf-8')

    # ------------------------------------------------------------------ calls
    def _call(self, job):
        try:
            res = self.client.complete(job['messages'])
            try:
                plan, assessment = parse_response(res['text'], job['id'], self.map)
                job.update(plan=plan, assessment=assessment)
            except PlanError as e:
                job['error'] = f'invalid plan: {e}'
            job.update(text=res['text'], usage=res.get('usage'), wall_s=res.get('wall_s'), model=res.get('model'),
                       reasoning_chars=res.get('reasoning_chars'))
        except Exception as e:                                   # API/network: no retry, recorded
            job['error'] = f'{type(e).__name__}: {e}'
            job['wall_s'] = round(time.perf_counter() - job['wall_start'], 3)
        job['wall_end'] = time.perf_counter()
        with self.lock:
            job['finished'] = True

    def submit(self, packet):
        user = compact_packet(packet, self.map)
        job = dict(id=packet['observation_id'], sent=packet['observation']['t'], trigger=packet['trigger'],
                   messages=[dict(role='system', content=self.system),
                             dict(role='user', content=json.dumps(user, ensure_ascii=False))],
                   wall_start=time.perf_counter(), finished=False)
        self.inflight = job
        if self.timing == 'pause':
            self._call(job)
        else:
            threading.Thread(target=self._call, args=(job,), daemon=True).start()

    def _deliverable(self, job, sim_t):
        if self.timing in ('pause', 'realtime'):
            return job['finished']
        if self.fixed is not None:
            if sim_t - job['sent'] + 1e-9 < self.fixed:
                return False
            while not job['finished']:
                time.sleep(0.005)
            return True
        # virtual: never let match time run ahead of wall time while waiting
        while True:
            with self.lock:
                finished = job['finished']
            if finished:
                return sim_t - job['sent'] + 1e-9 >= (job.get('wall_s') or 0.0)
            ahead = (sim_t - job['sent']) - (time.perf_counter() - job['wall_start'])
            if ahead <= 0:
                return False
            time.sleep(min(ahead, 0.05))

    def poll(self, sim_t):
        job = self.inflight
        if job is None or not self._deliverable(job, sim_t):
            return []
        self.inflight = None
        latency = round(sim_t - job['sent'], 3)
        rec = dict(observation_id=job['id'], sent_t=job['sent'], applied_t=sim_t, latency_s=latency,
                   wall_s=job.get('wall_s'), trigger=job['trigger'], plan=job.get('plan'),
                   assessment=job.get('assessment'), error=job.get('error'), usage=job.get('usage'),
                   model=job.get('model'), reasoning_chars=job.get('reasoning_chars'))
        self.calls.append(rec)
        if self.record_dir:
            n = f"{job['id']:03d}"
            (self.record_dir / f'{n}_request.json').write_text(json.dumps(
                dict(observation_id=job['id'], sent_t=job['sent'], trigger=job['trigger'],
                     client=self.client.public_config(), user=json.loads(job['messages'][1]['content'])),
                ensure_ascii=False, indent=1), encoding='utf-8')
            (self.record_dir / f'{n}_response.json').write_text(json.dumps(
                dict(rec, text=job.get('text')), ensure_ascii=False, indent=1), encoding='utf-8')
        return [dict(observation_id=job['id'], plan=job.get('plan'), assessment=job.get('assessment'),
                     error=job.get('error'), latency_s=latency, wall_s=job.get('wall_s'), usage=job.get('usage'))]
