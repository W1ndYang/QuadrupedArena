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

## First completed results (2026-11-05)

The 200k-step checkpoint completed training and held-out evaluation. On seeds
`33001..33020`, reproducibly sampled PPO scored 11.992 mean official points
versus 4.000 for fixed safe or fast capture; argmax PPO scored 4.302. The
checkpoint has 8,712 parameters. On a second 20-seed CPU ablation
(`35001..35020`), sampled PPO scored 12.184, while disabling subsequent
opponent visual updates scored 12.496. The learned benefit has not been shown
to depend on current opponent sighting.

One full Ubuntu Isaac trial on seed `34001` completed at 10 Hz simulation
cadence: 12.000 official points, 359 policy decisions and distinct RGB frames,
27 scene-derived opponent sightings, and no localization errors. The same
checkpoint and seed scored 12.000 in the CPU geometry-sensor evaluation. The
Isaac run took 398.2 wall-clock seconds for 35.88 simulation seconds; model
inference averaged 0.159 ms per decision, so the complete rendering/sensor
pipeline is not wall-clock real-time. This single online match establishes
execution and sensor integration, not an Isaac cross-seed performance estimate
or RGB-based opponent recognition.

## Record a video of the learned policy

On the Ubuntu Isaac machine, sync the current source and run:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
bash scripts/run_rl_skill_ubuntu_video.sh
```

The script uses the saved 200k checkpoint and match seed `34001` by default.
Pass a checkpoint path as its sole argument to use another model, or set
`RL_SKILL_POLICY_SEED` to change the match seed. It runs one live online match,
encodes the exact onboard Isaac RGB frames as `rl_skill_onboard.mp4`, then
re-renders the recorded 50 Hz match states in Isaac as a third-person
`rl_skill_overview.mp4`. The overview does not rerun the policy or referee;
its HUD and helper lines are presentation overlays. Both videos are 10 fps.
The existing renderer adds a short intro, outro and collapse presentation to
the overview, so its playback duration can exceed the match duration.

The terminal prints `VIDEO_OVERVIEW`, `VIDEO_ONBOARD` and `UPLOAD_THIS` paths.
The archive at `UPLOAD_THIS` contains both videos, the source match and replay
records, logs, source snapshots, checkpoint and a SHA-256 manifest. It also
packages diagnostics if either Isaac pass fails. The earlier completed online
match can already be encoded to an onboard video without rerunning Isaac:

```bash
python3 scripts/rl_skill_video.py \
  --archive ubuntu_logs/rl_skill_isaac_policy_trial_20261105-092650.tar.gz \
  --output .arena-workspace/experiments/rl-skill-video/previous_onboard.mp4
```
