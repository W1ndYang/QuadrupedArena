"""Public 10 Hz observations for the two-guard Skill policy.

The formal referee's packet may contain a precise defender yaw and a memory
updated on 50 Hz physics ticks. Neither is an input here. Only a sampled
visible position can update the policy's visual track.
"""
from collections import deque
import math

import numpy as np


CONTRACT_VERSION = 'two-guard-visual-history-v2'
SKILL_PHASES = ('approach', 'observe', 'retreat', 'advance')
FEEDBACK_NAMES = (
    'initial', 'ROUTE_ACTIVE', 'CONTINUOUS_ROUTE_ACTIVE',
    'NO_ROUTE_TO_REQUESTED_TARGET',
    'SMOKE_NOT_CURRENTLY_LEGAL_CONTINUE_DESTINATION',
)
FIRST = dict(cover=(-8., 0.), south_peek=(-8., -2.1),
             north_peek=(-8., 2.1), south_gate=(-5., -1.6),
             north_gate=(-5., 1.6), exit=(3., 0.), goal=(10., 0.))
SECOND = dict(cover=(3., 0.), south_peek=(3., -2.1),
              north_peek=(3., 2.1), south_gate=(6., -1.6),
              north_gate=(6., 1.6), exit=(10., 0.), goal=(10., 0.))
TARGET_NAMES = tuple(FIRST)
TRACK_TTL = 4.
VELOCITY_TTL = .5
VELOCITY_WINDOW = 1.


class GeometryDetectionAdapter:
    """Turn one legal referee detection into a sensor-shaped ground point."""
    def __init__(self, noise_std_m=0., miss_probability=0., seed=0):
        if noise_std_m < 0. or not 0. <= miss_probability <= 1.:
            raise ValueError('Invalid measurement noise or miss probability')
        self.noise_std_m = float(noise_std_m)
        self.miss_probability = float(miss_probability)
        self.rng = np.random.default_rng(seed)

    def sample(self, visible, frame_valid=True):
        if not frame_valid:
            return None
        if visible is None or self.rng.random() < self.miss_probability:
            return None
        # Intentionally ignore visible['yaw'] and every other referee field.
        point = np.asarray(visible['position'], dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError('Invalid visible ground position')
        if self.noise_std_m:
            point = point + self.rng.normal(0., self.noise_std_m, 2)
        return tuple(float(value) for value in point)


class VisualHistoryEncoder:
    def __init__(self, duration, action_count, sensor=None):
        if duration <= 0. or action_count <= 0:
            raise ValueError('Invalid duration or action catalog')
        self.duration = float(duration)
        self.action_count = int(action_count)
        self.sensor = sensor or GeometryDetectionAdapter()
        self.last_time = None
        self.tracks = (deque(maxlen=40), deque(maxlen=40))
        self.last_details = None
        names = [
            'self_x', 'self_y', 'body_yaw_sin', 'body_yaw_cos', 'self_speed',
            'camera_pan', 'remaining_time', 'posture_low',
            'posture_transition', 'smoke_available', 'smoke_remaining',
            'collision_stop', 'last_commit_age', 'stage_second',
        ]
        for target in TARGET_NAMES:
            names.extend((target + '_dx', target + '_dy'))
        names.extend((
            'frame_valid', 'detected_now', 'measured_dx', 'measured_dy',
            'measurement_uncertainty', 'track_valid', 'last_seen_dx',
            'last_seen_dy', 'last_seen_age', 'estimated_vx', 'estimated_vy',
            'velocity_valid', 'recent_detection_count',
        ))
        names.extend('feedback_' + name for name in FEEDBACK_NAMES)
        names.append('feedback_other')
        names.extend('skill_phase_' + name for name in SKILL_PHASES)
        names.append('skill_phase_other')
        names.extend(f'active_action_{i}' for i in range(action_count))
        names.extend(f'last_action_{i}' for i in range(action_count))
        self.names = tuple(names)
        self.index = {name: i for i, name in enumerate(self.names)}

    def _check_time(self, t):
        if not math.isfinite(t) or abs(t * 10. - round(t * 10.)) > 1e-5:
            raise ValueError('Observation must be sampled on the 10 Hz lattice')
        if self.last_time is not None and not math.isclose(
                t - self.last_time, .1, abs_tol=1e-5):
            raise ValueError('Visual history must advance exactly one 10 Hz step')
        if self.last_time is None and not math.isclose(t, 0., abs_tol=1e-5):
            raise ValueError('First observation must be at episode time zero')
        self.last_time = t

    @staticmethod
    def _velocity(recent):
        if len(recent) < 2:
            return (0., 0.), False
        times = np.asarray([row[0] for row in recent], dtype=float)
        points = np.asarray([row[1] for row in recent], dtype=float)
        centered = times - times.mean()
        denominator = float(centered @ centered)
        if denominator < 1e-8:
            return (0., 0.), False
        velocity = (centered @ (points - points.mean(axis=0))) / denominator
        return tuple(float(value) for value in velocity), True

    def encode(self, packet, stage, current_action, last_action,
               last_commit_time, frame_valid=True, skill_phase='idle'):
        if stage not in (0, 1):
            raise ValueError('Public map stage must be 0 or 1')
        t = float(packet['time'])
        self._check_time(t)
        own = packet['self']
        xy = np.asarray(own['position'], dtype=float)
        yaw = float(own['yaw'])
        if xy.shape != (2,) or not np.isfinite(xy).all() or not math.isfinite(yaw):
            raise ValueError('Invalid robot pose')
        point = self.sensor.sample(packet.get('visible'), frame_valid)
        track = self.tracks[stage]
        if point is not None:
            track.append((t, point))
        recent = [row for row in track if t - row[0] <= VELOCITY_WINDOW + 1e-8]
        count = sum(t - row[0] <= 2. + 1e-8 for row in track)
        last = track[-1] if track else None
        age = float('inf') if last is None else max(0., t - last[0])
        track_valid = age <= TRACK_TTL
        velocity, velocity_available = self._velocity(recent)
        velocity_valid = bool(velocity_available and age <= VELOCITY_TTL)
        if not velocity_valid:
            velocity = (0., 0.)
        last_xy = np.asarray(last[1]) if track_valid else xy
        measured = np.asarray(point) if point is not None else xy
        targets = FIRST if stage == 0 else SECOND
        values = [
            xy[0] / 12., xy[1] / 8., math.sin(yaw), math.cos(yaw),
            float(own['speed']) / 3.2,
            float(own.get('camera_pan', 0.)) / math.pi,
            max(0., self.duration - t) / self.duration,
            float(own.get('posture', 'upright') == 'low'),
            float(own.get('transition_remaining', 0.)) / .4,
            float(bool(packet.get('smoke_available', False))),
            float(own.get('smoke_remaining', 0.)),
            min(float(own.get('collision_stop_ticks', 0)), 100.) / 100.,
            min(max(0., t - float(last_commit_time)), 5.) / 5.,
            float(stage == 1),
        ]
        for name in TARGET_NAMES:
            delta = np.asarray(targets[name]) - xy
            values.extend((delta[0] / 24., delta[1] / 16.))
        values.extend((
            float(bool(frame_valid)), float(point is not None),
            (measured[0] - xy[0]) / 24., (measured[1] - xy[1]) / 16.,
            self.sensor.noise_std_m / 2.,
            float(track_valid),
            (last_xy[0] - xy[0]) / 24., (last_xy[1] - xy[1]) / 16.,
            min(age, TRACK_TTL) / TRACK_TTL,
            velocity[0] / 1.5, velocity[1] / 1.5,
            float(velocity_valid), min(count, 20) / 20.,
        ))
        feedback = own.get('execution_feedback', 'initial')
        values.extend(float(feedback == name) for name in FEEDBACK_NAMES)
        values.append(float(feedback not in FEEDBACK_NAMES))
        values.extend(float(skill_phase == name) for name in SKILL_PHASES)
        values.append(float(skill_phase not in SKILL_PHASES))
        values.extend(float(current_action == i) for i in range(self.action_count))
        values.extend(float(last_action == i) for i in range(self.action_count))
        result = np.asarray(values, dtype=np.float32)
        if len(result) != len(self.names) or not np.isfinite(result).all():
            raise RuntimeError('Non-finite or mismatched public observation')
        self.last_details = dict(
            time=t, stage=stage, frame_valid=bool(frame_valid),
            detected_now=point is not None, track_valid=track_valid,
            last_seen_age=None if last is None else age,
            velocity_valid=velocity_valid, recent_detection_count=count,
        )
        return result
