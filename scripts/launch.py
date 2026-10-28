"""Run with the existing pinned Isaac interpreter; keep all output outside source."""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from arena_paths import workspace_root, dependency_path

project = Path(__file__).resolve().parents[1]
workspace = workspace_root()
if "--help" in sys.argv:
    print("Usage: bash run.sh [--headless] [--empty] [--capture] [--robot go2] [--steps 240] [--interactive]")
    print("Default: finite GPU physics smoke test. --capture saves a 640x480 overview.")
    print("Terrain: --terrain flat|undulating|gravel|rough (default flat); --terrain-config PATH")
    print("--beauty: one 1280x960 Isaac path-traced image with PBR materials and hidden test objects.")
    print("--demo: scripted match GIF/MP4; --preview renders only five checkpoints.")
    print("--demo --tactical: Go2-W reactive cover demo; --preview renders ten checkpoints.")
    print("--demo --rules-v3: R0 rules trial, flat arena and deterministic wall collapse.")
    print("--demo --rules-v3 --adaptive: visibility-limited humanoid and risk-aware quadruped.")
    print("--demo --rotating-targets: single target, 5s from first entry, cyclic refresh; faster humanoid turns.")
    print("--art: industrial courtyard PBR scene + four native Isaac stills; --preview for quick renders.")
    print("--dockyard: 32x24m native Isaac map review, USD + stills + camera tour; no match. --preview / --no-tour optional.")
    print("--demo --replay-dir PATH --industrial: replay an existing finished match in the PBR courtyard; no policy rerun.")
    print("--embodied-probe --headless: 17s actual Go2-W joint-drive and onboard RGB preflight; no match.")
    print("--embodied-probe --manual-test --headless: three paused 2s local skills using actions/NNN.json; static target, not combat.")
    print("--vision-trial --headless: kinematic match with online Isaac RGB; waits for external actions/NNN.json every 2s.")
    print("--vision-trial --vision-smoke --headless: separate 2s camera sweep engineering test, not an LLM match.")
    print("--vision-trial --semantic-skills --headless: five robot intent skills and short four-field external decisions.")
    print("--vision-trial --semantic-skills --g1b1-visible-spawn --headless: G1-B.1 diagnostic A spawn facing B; inspect real RGB before annotating.")
    print("--vision-trial --geometry-diagnostic --headless: capture two formal-scene geometry samples, bundle them, and exit; no LLM or actions JSON.")
    print("  --diagnostic-output-dir PATH: optional diagnostic artifact directory; default is the generated run/geometry_diagnostic.")
    print("--vision-trial --geometry-interface-probe --headless: read-only formal-scene segmentation/depth interface probe; no Skill commit or asset writeback.")
    print("--vision-trial --semantic-skills --synthetic-vision-mvp --headless: one current-scene synthetic foot sighting, Skill handoff, then exit; no external action or API.")
    print("--vision-trial --semantic-skills --deepseek --headless: paid official DeepSeek Flash RGB planner, effort from config.")
    print("--vision-trial --semantic-skills --qwen-vision max|min --headless: paid Qwen3.6-27B public RGB; default max thinking budget or thinking off.")
    print("--vision-trial --semantic-skills --assessment --headless: external assistant, same four-field decision/assessment schema.")
    raise SystemExit(0)

if "--geometry-diagnostic" in sys.argv and "--vision-trial" not in sys.argv:
    raise SystemExit("--geometry-diagnostic requires --vision-trial")
if "--geometry-interface-probe" in sys.argv and "--vision-trial" not in sys.argv:
    raise SystemExit("--geometry-interface-probe requires --vision-trial")
if "--synthetic-vision-mvp" in sys.argv and not {"--vision-trial", "--semantic-skills"}.issubset(sys.argv):
    raise SystemExit("--synthetic-vision-mvp requires --vision-trial --semantic-skills")

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], capture_output=True, text=True)
if gpu.returncode:
    raise SystemExit("GPU inaccessible. Run on the host with GPU device access.\n" + gpu.stderr + gpu.stdout)
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
run_dir = workspace / "experiments/runs/quadruped-arena" / stamp
run_dir.mkdir(parents=True)
source_dir = run_dir / "source_snapshot"
for source in [project / "run.sh", *sorted((project / "scripts").glob("*.py")), *sorted((project / "configs").glob("*.json"))]:
    destination = source_dir / source.relative_to(project)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
repos = {}
for name in ("IsaacLab", "unitree_rl_lab", "unitree_ros"):
    repo = dependency_path(name)
    probe = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    if probe.returncode:
        repos[name] = {"path": str(repo), "available": repo.exists(), "git_checkout": False}
        continue
    head = probe.stdout.strip()
    diff = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD"])
    (run_dir / f"{name}.patch").write_bytes(diff)
    repos[name] = {"path": str(repo), "commit": head, "diff_sha256": hashlib.sha256(diff).hexdigest()}
manifest = {"python": sys.version, "executable": sys.executable, "packages": packages,
            "repositories": repos, "arguments": sys.argv[1:]}
(run_dir / "environment.json").write_text(json.dumps(manifest, indent=2)+"\n")
entrypoint = "embodied_probe.py" if "--embodied-probe" in sys.argv else "art_scene.py" if "--art" in sys.argv else "trial_demo.py" if any(flag in sys.argv for flag in ('--rules-v3','--rotating-targets','--replay-dir')) else "demo.py" if "--demo" in sys.argv else "scene.py"
timeout = 1200 if "--demo" in sys.argv else 600 if any(flag in sys.argv for flag in ('--art','--embodied-probe')) else 300
if '--dockyard' in sys.argv:entrypoint='dockyard_scene.py';timeout=2400
if '--vision-trial' in sys.argv:entrypoint='vision_trial.py';timeout=5400
if '--geometry-diagnostic' in sys.argv:timeout=1200
if '--geometry-interface-probe' in sys.argv:timeout=180
if '--synthetic-vision-mvp' in sys.argv:timeout=180
if '--vision-trial' in sys.argv and '--deepseek' in sys.argv:
    from deepseek_trial import settings
    timeout=settings().get('launch_timeout_seconds',timeout)
command = [sys.executable, str(project / "scripts" / entrypoint), "--run-dir", str(run_dir), *sys.argv[1:]]
print(f"GPU: {gpu.stdout.strip()}\nRun: {run_dir}", flush=True)
print(f"Log: {run_dir / 'runtime.log'}", flush=True)
with (run_dir / "runtime.log").open("w") as log:
    log.write(gpu.stdout)
    log.flush()
    try:
        result = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT,
                                env=os.environ.copy(), timeout=None if "--interactive" in sys.argv else timeout)
    except subprocess.TimeoutExpired:
        (run_dir / "process.json").write_text(json.dumps({"status": "timeout", "limit_seconds": timeout})+"\n")
        raise SystemExit(f"Isaac exceeded {timeout} seconds. Inspect {run_dir / 'runtime.log'}")
tail = (run_dir / "runtime.log").read_text(errors="replace").splitlines()[-20:]
print("\n".join(tail))
print(f"Result: exit={result.returncode}; {run_dir}", flush=True)
result_path = run_dir / "result.json"
passed = result.returncode == 0 and result_path.exists() and json.loads(result_path.read_text())["status"] == "passed"
if result_path.exists():
    print(result_path.read_text(), flush=True)
(run_dir / "process.json").write_text(json.dumps({"exit_code": result.returncode, "passed": passed})+"\n")
raise SystemExit(0 if passed else 1)
