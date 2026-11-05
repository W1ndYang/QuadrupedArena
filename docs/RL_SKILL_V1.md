# 10 Hz Skill RL v1 contract

This is the implementation contract for the first learned policy. It does not
claim that geometric simulator perception is an RGB detector or a real-robot
sensor.

## Clocks and boundary

- Referee, motion and Skill guards advance at 0.02 s (50 Hz).
- The actor observes and selects an action every 0.1 s (10 Hz).
- The simulator-native perception sensor samples every 0.1 s. It applies range,
  onboard-camera field of view, map occlusion, dropout and position noise before
  updating `VisualMemory`.
- Opponent position and hidden policy state are confined to the sensor adapter,
  reward and evaluator. They are not members of the actor vector.
- This sensor is an explicitly declared geometric simulator interface. The
  existing RGB foot-pixel path remains separate.

## Actor observation

The normalized flat vector contains own pose, signed speed, HP and cooldown;
official time, score and rotating-target state; relative active-target geometry;
collapse time; camera pan; timestamped visual belief, age and uncertainty;
current semantic Skill/status; previous compiled low-level action and execution
feedback; and the previous policy action. It contains no opponent truth, referee
hazard threshold, opponent policy state or future rollout.

The v1 critic uses the same observation. Asymmetric privileged critic input is
left out of the first version to keep leakage auditing and NumPy inference
simple.

## Action catalog and execution semantics

| Index | Policy action | Formal request |
|---:|---|---|
| 0 | `CONTINUE` | Preserve the current Skill and route |
| 1 | `CAPTURE_POINT_SAFE` | `CAPTURE_POINT(active, safe)` |
| 2 | `CAPTURE_POINT_FAST` | `CAPTURE_POINT(active, fast)` |
| 3 | `TAKE_COVER_SAFE` | `TAKE_COVER(safe)` |
| 4 | `AIM_AT_TRACK` | `AIM_AT_TRACK` |
| 5 | `OBSERVE_SCAN` | `OBSERVE(scan)` |
| 6 | `OBSERVE_THREAT` | `OBSERVE(threat)` |

Selecting the same parameterized intent again means continue, not recommit. A
target-generation change invalidates the signature and forces a new formal
commit. Selecting another intent may interrupt the current Skill. Every step
reports requested, validated, committed/continued, compiled low-level action,
status and blocked reason. `MOVE_TO` is omitted because moving to a task point is
already represented by `CAPTURE_POINT`; cover choice remains inside the formal
risk-aware Skill planner.

## Objective and evaluation

PPO reward keeps official score as the largest positive term and separately
logs score delta, HP changes, target progress, occupancy, blocked requests,
switch cost and survival. The switch cost is deliberately material enough to
discourage stochastic Skill thrashing while leaving official score dominant.
Evaluation never uses shaping: it compares official
full-match score over held-out seeds against fixed safe capture, fast capture,
and observe-then-capture baselines. Both argmax and reproducibly sampled PPO
deployment are reported. Score ties remain ties.

Seeds `31001..31020` are the development/validation set. Seeds
`32001..32020` were the first frozen test for the 40k checkpoint. Because the
decision to continue training was made after viewing that result, a new untouched
set, `33001..33020`, is the evaluation command's default for the 200k run. Do not
tune a later checkpoint on the new frozen results.

The implementation uses a one-hidden-layer NumPy PPO actor and critic, saves
optimizer state for resume, and exports a compressed `.npz`. Training and
evaluation logs remain under `.arena-workspace/`; source control contains only
code and configuration.

Run the complete CPU training, held-out evaluation and packaging workflow with:

```bash
bash scripts/run_rl_skill_v1.sh
```

The default 200,000 decision steps are expected to take roughly one to two
hours on a contemporary CPU; actual time is recorded. A short engineering
smoke test can use `rl_skill_train.py --total-steps 256 --rollout-steps 128`.

On the Ubuntu Isaac machine, run the 10 Hz sensor validation and automatic
packaging with:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
bash scripts/run_rl_skill_ubuntu_validation.sh
```

The wrapper packages artifacts even when the Isaac validation fails. Return the
single path printed as `UPLOAD_THIS`; its archive contains the runtime log,
result, RGB frames, per-sample diagnostics, source snapshot and SHA-256 manifest.

For one complete match with the 200k checkpoint and live Isaac scene-derived
perception, run on Ubuntu:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
bash scripts/run_rl_skill_ubuntu_policy_trial.sh \
  "$PWD/.arena-workspace/experiments/rl-skill-jobs/20261104-153934/train/checkpoints/final.npz"
```

The script defaults to match seed `34001` and packages the checkpoint, match
trace, 10 Hz JPEG frames, source snapshot, logs, and SHA-256 manifest even if
the trial fails. This test measures transfer from the CPU training sensor to
the Isaac scene-derived sighting chain; the game referee and motion remain the
same Python implementation.
