"""Call-and-return Skill execution over the same 10 Hz public sensor stream.

The high-level policy is called only at option boundaries. The existing formal
Skill executor still runs on every 50 Hz physics tick.
"""
import math

from online_env import OnlineController, run_episode as run_online_episode
from env import IntentController


WAIT_SECONDS = 1.
COLLISION_INTERRUPT_TICKS = 10
FAILURE_FEEDBACK = {
    'NO_ROUTE_TO_REQUESTED_TARGET',
    'SMOKE_NOT_CURRENTLY_LEGAL_CONTINUE_DESTINATION',
}
MOVING_ACTIONS = {1, 2, 3, 5, 6, 7, 8, 9}


class OptionController(OnlineController):
    def __init__(self, cfg, rules, waypoints, session):
        super().__init__(cfg, rules, waypoints, session)
        self.active_option = None
        self.pending_handoff = False
        self.previous_detected = False
        self.visual_tick_counts = [0, 0]
        self.visual_sample_counts = [0, 0]
        self.last_collision_count = None
        self.collision_streak = 0

    def reset_stage(self, t):
        super().reset_stage(t)
        # The previous option closes at the next 10 Hz decision boundary.
        self.pending_handoff = True
        self.previous_detected = False
        self.collision_streak = 0
        self.last_collision_count = self.session.probe.collision_stops[0]
        self.session.probe.feedback = 'initial'

    def legal_actions(self, packet):
        mask = super().legal_actions(packet)
        mask[0] = False  # CONTINUE has no meaning at a returned boundary.
        cover_x = -8. if self.session.stage == 0 else 3.
        if math.dist(packet['self']['position'], (cover_x, 0.)) < .15:
            mask[1] = False
        # Re-selecting a completed peek creates a fresh option and reissues it.
        if not self.passed_gate:
            mask[2:4] = True
        return mask

    def _termination_reason(self, packet, t, detected):
        option = self.active_option
        if option is None:
            return None
        if self.pending_handoff:
            return 'stage_handoff'
        if t - option['time'] < .1 - 1e-6:
            return None
        action = option['action']
        own = packet['self']
        if (action in MOVING_ACTIONS and
                own.get('execution_feedback') in FAILURE_FEEDBACK):
            return 'interrupted_execution_failure'
        if (action in MOVING_ACTIONS and
                self.collision_streak >= COLLISION_INTERRUPT_TICKS):
            return 'interrupted_collision'
        if action == 4:
            if detected and not self.previous_detected:
                return 'interrupted_new_detection'
            if t - option['time'] >= WAIT_SECONDS - 1e-6:
                return 'wait_elapsed'
        if action == 1:
            cover_x = -8. if self.session.stage == 0 else 3.
            if math.dist(own['position'], (cover_x, 0.)) < .15:
                return 'cover_reached'
        if action in (2, 3):
            cover_x = -8. if self.session.stage == 0 else 3.
            if (self.phase == 'retreat' and
                    math.dist(own['position'], (cover_x, 0.)) < .15):
                return 'peek_returned'
        if action in (5, 6, 8, 9) and self.passed_gate:
            return 'gate_crossed'
        return None

    def _finish_option(self, t, reason):
        option = self.active_option
        if option is None:
            return
        elapsed = max(0., float(t) - option['time'])
        option['end_time'] = float(t)
        option['elapsed_seconds'] = elapsed
        option['duration_steps'] = elapsed / .1
        option['termination_reason'] = reason
        self.active_option = None
        self.pending_handoff = False

    def finalize_options(self, terminal_time):
        self._finish_option(terminal_time, 'episode_terminal')

    def decide(self, packet):
        t = float(packet['time'])
        collisions = int(packet['self'].get('collision_stop_ticks', 0))
        if self.last_collision_count is not None:
            self.collision_streak = (self.collision_streak + 1
                                     if collisions > self.last_collision_count else 0)
        self.last_collision_count = collisions
        if round(t / self.rules['dt']) % 5 == 0:
            mask = self.legal_actions(packet)
            observation = self.encoder.encode(
                packet, self.session.stage, self.current_intent_action,
                self.last_action, self.last_commit_time,
                skill_phase=self.phase)
            stage = self.session.stage
            self.visual_sample_counts[stage] += 1
            detected = bool(self.encoder.last_details['detected_now'])
            self.visual_tick_counts[stage] += int(detected)
            observer = getattr(self.session.policy, 'observe', None)
            if observer is not None:
                observer(observation, mask.copy())
            reason = self._termination_reason(packet, t, detected)
            if self.active_option is None or reason is not None:
                if reason is not None:
                    self._finish_option(t, reason)
                if reason=='cover_reached' and self.passed_gate:
                    self.passed_gate=False
                    self.chosen_gate=None
                    mask=self.legal_actions(packet)
                action, logp, value = self.session.act(observation, mask.copy())
                action = int(action)
                if not 0 <= action < len(mask) or not mask[action]:
                    raise ValueError(f'Option policy selected illegal action {action}')
                record = dict(
                    time=t, stage=stage, observation=observation,
                    action_mask=mask, action=action, logp=float(logp),
                    value=float(value), own_xy=list(packet['self']['position']),
                    visual=self.encoder.last_details.copy())
                self.records.append(record)
                self.active_option = record
                # Even the same action is a new invocation after return.
                self.intent = dict(observation_id=len(self.records) - 1,
                                   **self.request(action, packet))
                self.current_intent_action = action
                self.phase = 'approach'
                self.hold_started = None
                self.tag_done = False
                self.last_commit_time = t
                self.last_action = action
                self.collision_streak = 0
                # The referee's feedback is latched; it belongs to the option
                # that just returned, not to this newly committed Skill.
                self.session.probe.feedback = 'initial'
            self.previous_detected = detected
            self.samples = []
        return IntentController.decide(self, packet)


def run_episode(first_seed, second_seed, policy, duration=48., sensor=None):
    summary, records = run_online_episode(
        first_seed, second_seed, policy, duration=duration, sensor=sensor,
        controller_class=OptionController)
    if any('duration_steps' not in row or row['duration_steps'] <= 0.
           for row in records):
        raise RuntimeError('An option lacks a positive duration')
    return summary, records
