"""Collect one existing Isaac Oracle smoke attempt and a self-contained upload archive.

Only the shell wrapper is the user entry point. Constructor injection is used by
CPU tests; the production command and repository are fixed in main().
"""

from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import urllib.parse
import uuid


DEFAULT_SMOKE = ("bash", "run.sh", "--vision-trial", "--semantic-skills",
                 "--synthetic-vision-skill-runtime-smoke", "--oracle-state-diagnostic", "--headless")
TEST_COMMANDS = (
    ("oracle", ("python3", "-m", "unittest", "scripts.test_oracle_state_v1", "-v")),
    ("related", ("python3", "-m", "unittest", "scripts.test_trial_rules",
                 "scripts.test_semantic_skills", "scripts.test_rotating_targets",
                 "scripts.test_vision_core", "scripts.test_synthetic_vision_mvp",
                 "scripts.test_adaptive_motion", "-v")),
    ("diff_check", ("git", "diff", "--check")),
)
KEY_FILES = ("run.sh", "scripts/launch.py", "scripts/vision_trial.py",
             "scripts/oracle_state_v1.py", "scripts/oracle_state_v1_smoke.py",
             "scripts/collect_oracle_v1_1_runtime_smoke.sh",
             "scripts/collect_oracle_v1_1_runtime_smoke.py",
             "configs/trial_rotating.json", "configs/trial_adaptive.json",
             "configs/trial_v3.json")
RUN_FILES = {
    "oracle_state_diagnostic.json": "diagnostics/oracle_state_diagnostic.json",
    "synthetic_vision_skill_runtime_smoke.json": "diagnostics/synthetic_vision_skill_runtime_smoke.json",
    "runtime.log": "logs/isaac_runtime.log",
    "environment.json": "environment/launch_environment.json",
    "process.json": "diagnostics/process.json",
    "result.json": "diagnostics/result.json",
    "protocol.json": "diagnostics/protocol.json",
    "frame_index.json": "diagnostics/frame_index.json",
    "interruption.json": "diagnostics/interruption.json",
    "skill_execution.json": "diagnostics/skill_execution.json",
}
RUN_JSON_GLOBS = ("synthetic_vision_windows/*.json", "synthetic_vision_frames/*.json",
                  "observations/*.json", "observations/frames/*.json",
                  "decisions/*.json", "actions/*.json")


def timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_remote(url: str) -> dict[str, str]:
    """Keep only host and last repository name; discard credentials and query."""
    value = url.strip()
    if "://" in value:
        parsed = urllib.parse.urlsplit(value)
        host = parsed.hostname or "unknown"
        repo = Path(parsed.path.rstrip("/")).name
        scheme = parsed.scheme
    elif ":" in value and not value.startswith("/"):
        left, path = value.split(":", 1)
        host = left.rsplit("@", 1)[-1]
        repo = Path(path.rstrip("/")).name
        scheme = "ssh"
    else:
        host, repo, scheme = "local_path", Path(value).name, "local"
    # Even a malicious repository basename should not put a token in an upload.
    repo = re.sub(r"[^A-Za-z0-9_.-]", "_", repo)[:80]
    host = re.sub(r"[^A-Za-z0-9_.-]", "_", host)[:120]
    return {"scheme": scheme, "host": host, "repository_basename": repo}


def command_string(command: tuple[str, ...] | list[str], env_prefix: str = "") -> str:
    import shlex
    return env_prefix + " ".join(shlex.quote(part) for part in command)


class Collector:
    def __init__(self, repo: Path, output_root: Path, *, smoke_command=DEFAULT_SMOKE,
                 test_commands=TEST_COMMANDS, timeout_seconds: int = 1800,
                 runtime_parent: Path | None = None):
        self.repo = repo.resolve()
        self.output_root = output_root.expanduser().resolve()
        if self.output_root == self.repo or self.repo in self.output_root.parents:
            raise ValueError("Output root must be outside the Git repository")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_id = "oracle_v1_1_runtime_smoke_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.directory = self.output_root / self.run_id
        self.directory.mkdir(mode=0o700)
        for name in ("environment", "logs", "diagnostics", "tests", "summary"):
            (self.directory / name).mkdir()
        self.archive = self.output_root / (self.run_id + ".tar.gz")
        self.smoke_command = tuple(smoke_command)
        self.test_commands = tuple((name, tuple(command)) for name, command in test_commands)
        if type(timeout_seconds) is not int or timeout_seconds < 1:
            raise ValueError("Timeout must be a positive integer")
        self.timeout_seconds = timeout_seconds
        workspace = Path(os.environ.get("ARENA_WORKSPACE", self.repo / ".arena-workspace")).expanduser().resolve()
        self.runtime_parent = (runtime_parent or workspace / "experiments/runs/quadruped-arena").resolve()
        self.started_epoch = time.time()
        self.started_at = timestamp()
        self.active_process: subprocess.Popen | None = None
        self.interrupted = False
        self.command_records: list[dict] = []
        self.anomalies: list[str] = []
        self.smoke: dict = {"status": "not_run", "exit_code": None, "runtime_dir": None}
        self.tests: dict[str, dict] = {}
        self.diagnostic: dict = {"found": False, "copied_path": None, "boundaries": [], "parse_error": None}
        self.environment: dict = {}

    def _signal(self, signum, _frame):
        self.interrupted = True
        self.anomalies.append("received_signal_" + signal.Signals(signum).name)
        self._stop_active()

    def _stop_active(self):
        process = self.active_process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def _run(self, name: str, command: tuple[str, ...], log_path: Path,
             *, timeout: int, echo: bool = False, env_prefix: str = "",
             env: dict[str, str] | None = None) -> dict:
        started = timestamp()
        t0 = time.monotonic()
        record = {"name": name, "command": command_string(command, env_prefix),
                  "started_at": started, "ended_at": None, "duration_seconds": None,
                  "exit_code": None, "status": "not_run", "log": str(log_path.relative_to(self.directory))}
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log:
            log.write(("Command: " + record["command"] + "\nStarted: " + started + "\n").encode())
            log.flush()
            try:
                process = subprocess.Popen(command, cwd=self.repo, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                self.active_process = process
                def pump():
                    assert process.stdout is not None
                    while True:
                        block = os.read(process.stdout.fileno(), 65536)
                        if not block:
                            break
                        log.write(block)
                        log.flush()
                        if echo:
                            sys.stdout.buffer.write(block)
                            sys.stdout.buffer.flush()
                reader = threading.Thread(target=pump, daemon=True)
                reader.start()
                try:
                    return_code = process.wait(timeout=timeout)
                    record["status"] = "interrupted" if self.interrupted else ("success" if return_code == 0 else "failed")
                except subprocess.TimeoutExpired:
                    record["status"] = "timeout"
                    self._stop_active()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    return_code = 124
                reader.join(timeout=10)
                if reader.is_alive():
                    self.anomalies.append(name + "_log_reader_still_active")
                elif process.stdout is not None:
                    process.stdout.close()
                record["exit_code"] = return_code
            except (OSError, ValueError) as exc:
                log.write((f"\nCollector could not start command: {type(exc).__name__}: {exc}\n").encode())
                record["status"] = "not_run"
                self.anomalies.append(name + "_command_not_started")
            finally:
                self.active_process = None
                record["ended_at"] = timestamp()
                record["duration_seconds"] = round(time.monotonic() - t0, 3)
                log.write(("\nEnded: " + record["ended_at"] + "\nStatus: " + record["status"] +
                           "\nExit code: " + str(record["exit_code"]) + "\n").encode())
        self.command_records.append(record)
        return record

    def _git(self, *args: str) -> tuple[int, str]:
        try:
            result = subprocess.run(("git", *args), cwd=self.repo, capture_output=True,
                                    text=True, timeout=15)
            return result.returncode, result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return 127, ""

    def capture_environment(self):
        env_dir = self.directory / "environment"
        remote_code, remote_names = self._git("remote")
        remotes = {}
        if remote_code == 0:
            for name in remote_names.splitlines():
                code, url = self._git("remote", "get-url", name)
                remotes[name] = safe_remote(url) if code == 0 else {"error": "unavailable"}
        _, branch = self._git("branch", "--show-current")
        _, head = self._git("rev-parse", "HEAD")
        status_code, status = self._git("status", "--short")
        (env_dir / "git_status.txt").write_text(status + "\n")
        hashes = {}
        config_dir = env_dir / "configs"
        config_dir.mkdir()
        for name in KEY_FILES:
            path = self.repo / name
            hashes[name] = sha256_file(path) if path.is_file() else None
            if name.startswith("configs/") and path.is_file():
                shutil.copy2(path, config_dir / path.name)
        code = {"branch": branch, "head": head, "remotes": remotes,
                "git_status_exit_code": status_code, "uncommitted_changes": bool(status.strip()),
                "key_file_sha256": hashes}
        (env_dir / "code.json").write_text(json.dumps(code, indent=2, ensure_ascii=False) + "\n")
        whitelisted = {name: os.environ[name] for name in
                       ("CONDA_PREFIX", "VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES", "ARENA_PYTHON", "ARENA_WORKSPACE")
                       if name in os.environ}
        host = {"collected_at": timestamp(), "collector_cwd": os.getcwd(), "repo_root": str(self.repo),
                "user": getpass.getuser(),
                "shell": os.environ.get("SHELL"), "uname": list(os.uname()),
                "collector_python": {"executable": sys.executable, "version": sys.version},
                "allowed_environment": whitelisted,
                "nvidia_eula_accepted": os.environ.get("OMNI_KIT_ACCEPT_EULA") == "YES"}
        (env_dir / "host.json").write_text(json.dumps(host, indent=2, ensure_ascii=False) + "\n")
        os_release = Path("/etc/os-release")
        (env_dir / "os_release.txt").write_text(os_release.read_text(errors="replace") if os_release.exists() else "unavailable\n")
        try:
            gpu = subprocess.run(("nvidia-smi",), capture_output=True, text=True, timeout=20)
            gpu_text = gpu.stdout + gpu.stderr
            gpu_info = {"available": gpu.returncode == 0, "exit_code": gpu.returncode}
        except (OSError, subprocess.TimeoutExpired) as exc:
            gpu_text = f"nvidia-smi unavailable: {type(exc).__name__}: {exc}\n"
            gpu_info = {"available": False, "exit_code": None}
        (env_dir / "nvidia_smi.txt").write_text(gpu_text)
        help_result = self._run("run_help", ("bash", "run.sh", "--help"), env_dir / "run_help.txt", timeout=30)
        shell_result = self._run("run_shell_syntax", ("bash", "-n", "run.sh"),
                                 env_dir / "run_shell_syntax.txt", timeout=10)
        expected = all(token in (self.repo / "scripts/vision_trial.py").read_text() for token in
                       ("--oracle-state-diagnostic", "--synthetic-vision-skill-runtime-smoke"))
        expected &= "--oracle-state-diagnostic" in (self.repo / "scripts/launch.py").read_text()
        cli = {"run_help_exit_code": help_result["exit_code"],
               "run_shell_syntax_exit_code": shell_result["exit_code"],
               "expected_flags_present_in_source": expected,
               "smoke_command": command_string(self.smoke_command)}
        (env_dir / "cli_check.json").write_text(json.dumps(cli, indent=2) + "\n")
        runtime_python = os.environ.get("ARENA_PYTHON", "python")
        runtime_python_available = shutil.which(runtime_python) is not None
        probe_python = runtime_python if runtime_python_available else sys.executable
        probe = ("import importlib.util, importlib.metadata, json, sys; "
                 "names=['isaaclab','isaacsim','omni','numpy','PIL']; out={}; "
                 "\nfor name in names:\n"
                 " try:\n  present=importlib.util.find_spec(name) is not None\n"
                 "  dist={'PIL':'Pillow'}.get(name,name)\n"
                 "  try: version=importlib.metadata.version(dist)\n"
                 "  except importlib.metadata.PackageNotFoundError: version=None\n"
                 "  out[name]={'import_spec_present':present,'distribution_version':version}\n"
                 " except Exception as exc: out[name]={'error':type(exc).__name__+': '+str(exc)}\n"
                 "print(json.dumps({'executable':sys.executable,'version':sys.version,'modules':out}))")
        dep_record = self._run("dependency_probe", (probe_python, "-c", probe),
                               env_dir / "dependency_probe.log", timeout=30)
        dependencies = {"command_status": dep_record["status"], "python_command": runtime_python,
                        "runtime_python_command_available": runtime_python_available,
                        "probe_python_command": probe_python,
                        "modules": {}, "parse_error": None}
        if dep_record["exit_code"] == 0:
            try:
                lines = (env_dir / "dependency_probe.log").read_text().splitlines()
                dependencies.update(json.loads(next(line for line in lines if line.startswith("{"))))
            except (ValueError, StopIteration) as exc:
                dependencies["parse_error"] = str(exc)
        (env_dir / "dependencies.json").write_text(json.dumps(dependencies, indent=2) + "\n")
        self.environment = {"code": code, "host": host, "gpu": gpu_info,
                            "cli": cli, "dependencies": dependencies}

    def discover_runtime(self, smoke_log: Path) -> Path | None:
        lines = smoke_log.read_text(errors="replace").splitlines()
        paths = [line[5:].strip() for line in lines if line.startswith("Run: ")]
        if len(paths) != 1:
            self.anomalies.append("expected_exactly_one_Run_path_found_" + str(len(paths)))
            return None
        candidate = Path(paths[0]).expanduser().resolve()
        if candidate.parent != self.runtime_parent or not candidate.is_dir():
            self.anomalies.append("Run_path_not_a_current_expected_runtime_directory")
            return None
        if candidate.stat().st_mtime < self.started_epoch - 5:
            self.anomalies.append("Run_path_older_than_this_collection")
            return None
        return candidate

    def collect_runtime(self, runtime: Path | None):
        if runtime is None:
            self.diagnostic["searched_path"] = "No verified current Run: path in smoke log"
            self.diagnostic["search_method"] = "anchored Run: line; expected parent and creation-time check"
            self.diagnostic["search_command"] = "No path was trusted; scan only anchored Run: line in logs/smoke_stdout_stderr.log"
            self.anomalies.append("diagnostic_json_missing")
            return
        self.smoke["runtime_dir"] = str(runtime)
        def current_file(path: Path) -> bool:
            return (path.is_file() and not path.is_symlink() and runtime in path.resolve().parents
                    and path.stat().st_mtime >= self.started_epoch - 5)
        copied = []
        for source_name, target_name in RUN_FILES.items():
            source = runtime / source_name
            if current_file(source):
                destination = self.directory / target_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied.append(target_name)
        for pattern in RUN_JSON_GLOBS:
            for source in sorted(runtime.glob(pattern)):
                if current_file(source):
                    target = self.directory / "diagnostics" / "runtime" / source.relative_to(runtime)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    copied.append(str(target.relative_to(self.directory)))
        images = []
        for source in runtime.rglob("*"):
            if current_file(source) and source.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm"):
                images.append({"relative_path": str(source.relative_to(runtime)),
                               "size_bytes": source.stat().st_size,
                               "purpose": "visual evidence; omitted from compact state diagnostic bundle"})
        (self.directory / "diagnostics/image_inventory.json").write_text(json.dumps(images, indent=2) + "\n")
        (self.directory / "diagnostics/copied_files.json").write_text(json.dumps(copied, indent=2) + "\n")
        diagnostic_source = runtime / "oracle_state_diagnostic.json"
        self.diagnostic["searched_path"] = str(diagnostic_source)
        self.diagnostic["search_method"] = "exact file under verified current Run: directory"
        self.diagnostic["search_command"] = command_string(("test", "-f", str(diagnostic_source)))
        if not current_file(diagnostic_source):
            self.anomalies.append("diagnostic_json_missing")
            return
        self.diagnostic["found"] = True
        self.diagnostic["copied_path"] = "diagnostics/oracle_state_diagnostic.json"
        try:
            payload = json.loads(diagnostic_source.read_text())
            if "render_sync_quality" not in payload or "same_frame_verified" not in payload:
                raise ValueError("top-level sync quality fields are missing")
            boundaries = payload["boundaries"]
            if not isinstance(boundaries, list):
                raise ValueError("boundaries must be a list")
            out = []
            previous_time = -1.0
            for index, item in enumerate(boundaries):
                seconds = item["simulation_seconds"]
                count = item["frame_count"]
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or (index and seconds <= previous_time):
                    raise ValueError("boundary times are invalid or out of order")
                if type(count) is not int or count != (1 if index == 0 else 4):
                    raise ValueError("boundary frame count is inconsistent")
                previous_time = seconds
                full, limited = item["full"], item["limited"]
                for view_name, state in (("full", full), ("limited", limited)):
                    if not isinstance(state, dict) or state.get("status") not in ("valid", "blocked", "terminal"):
                        raise ValueError(view_name + " state status is missing/invalid")
                    if state["status"] == "valid" and not isinstance(state.get("state_id"), str):
                        raise ValueError(view_name + " valid state lacks ID")
                    if state["status"] == "blocked" and state.get("state_id") is not None:
                        raise ValueError(view_name + " blocked state unexpectedly has ID")
                if full.get("decision_step") != limited.get("decision_step"):
                    raise ValueError("view decision steps differ")
                out.append({"decision_step": full.get("decision_step"),
                            "simulation_seconds": seconds, "frame_count": count,
                            "map_phase": full.get("map_phase"),
                            "A.region_id": full.get("A.region_id"),
                            "B.region_id": full.get("B.region_id"),
                            "full": {k: full.get(k) for k in ("status", "reason", "state_id")},
                            "limited": {k: limited.get(k) for k in ("status", "reason", "state_id")},
                            "render_sync_quality": item.get("render_sync_quality")})
            self.diagnostic["boundaries"] = out
            self.diagnostic["boundary_count"] = len(out)
            self.diagnostic["render_sync_quality"] = payload.get("render_sync_quality")
            self.diagnostic["same_frame_verified"] = payload.get("same_frame_verified")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            self.diagnostic["parse_error"] = f"{type(exc).__name__}: {exc}"
            self.anomalies.append("diagnostic_json_parse_failed")

    def run_tests(self):
        if self.interrupted:
            for name, _ in self.test_commands:
                self.tests[name] = {"status": "not_run_interrupted", "exit_code": None}
            return
        for name, command in self.test_commands:
            env = os.environ.copy()
            prefix = ""
            if name in ("oracle", "related"):
                env["PYTHONPATH"] = "scripts"
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                prefix = "PYTHONPATH=scripts "
            record = self._run(name, command, self.directory / "tests" / f"{name}.log",
                               timeout=900, env_prefix=prefix, env=env)
            self.tests[name] = record
            if self.interrupted:
                break

    def finish(self) -> dict:
        if self.interrupted:
            self.smoke["status"] = "interrupted"
        elif self.smoke["status"] == "success" and (not self.diagnostic["found"] or self.diagnostic["parse_error"]):
            self.smoke["status"] = "failed"
        ended_at = timestamp()
        startup_evidence = self._isaac_startup_evidence()
        real_isaac_started = (True if startup_evidence else False if self.smoke["status"] == "not_run" or
                              self.smoke["runtime_dir"] is None else None)
        summary = {"run_id": self.run_id, "started_at": self.started_at,
                   "ended_at": ended_at, "duration_seconds": round(time.time() - self.started_epoch, 3),
                   "repo_root": str(self.repo), "output_directory": str(self.directory),
                   "smoke": self.smoke, "diagnostic": self.diagnostic,
                   "diagnostic_json_found": self.diagnostic["found"],
                   "tests": {name: {"status": item["status"], "exit_code": item["exit_code"],
                                     "log": item.get("log")} for name, item in self.tests.items()},
                   "dependency_probe": self.environment.get("dependencies", {}),
                   "uncommitted_changes": self.environment.get("code", {}).get("uncommitted_changes"),
                   "isaac_startup_evidence": startup_evidence,
                   "real_isaac_started": real_isaac_started,
                   "anomalies": self.anomalies,
                   "manifest_note": "manifest_files.txt lists all payload files except itself and manifest.sha256"}
        summary_dir = self.directory / "summary"
        (summary_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        lines = [f"Oracle v1.1 runtime smoke collection: {self.run_id}",
                 f"Started: {self.started_at}", f"Ended: {ended_at}",
                 f"Smoke: {self.smoke['status']} (exit={self.smoke['exit_code']})",
                 f"Isaac startup evidence: {summary['isaac_startup_evidence']}; actual start: {real_isaac_started}",
                 f"Diagnostic JSON found: {self.diagnostic['found']}",
                 f"Diagnostic boundaries: {len(self.diagnostic['boundaries'])}",
                 f"Render sync quality: {self.diagnostic.get('render_sync_quality')}",
                 f"Same frame verified: {self.diagnostic.get('same_frame_verified')}",
                 "Tests: " + ", ".join(f"{name}={item['status']}({item['exit_code']})" for name, item in self.tests.items()),
                 f"Uncommitted changes: {summary['uncommitted_changes']}",
                 "Anomalies: " + (", ".join(self.anomalies) or "none"),
                 "Upload the matching .tar.gz archive; no manual file selection is needed."]
        for boundary in self.diagnostic["boundaries"]:
            lines.append("Boundary: " + json.dumps(boundary, ensure_ascii=False, sort_keys=True))
        (summary_dir / "summary.txt").write_text("\n".join(lines) + "\n")
        (summary_dir / "commands_and_exit_codes.txt").write_text("\n\n".join(
            json.dumps(item, ensure_ascii=False, indent=2) for item in self.command_records) + "\n")
        manifest = summary_dir / "manifest_files.txt"
        entries = []
        for path in sorted(self.directory.rglob("*")):
            if path.is_file() and path.name not in ("manifest_files.txt", "manifest.sha256"):
                entries.append(f"{path.relative_to(self.directory)}\t{path.stat().st_size}\t{sha256_file(path)}")
        manifest.write_text("relative_path\tsize_bytes\tsha256\n" + "\n".join(entries) + "\n")
        (summary_dir / "manifest.sha256").write_text(
            sha256_file(manifest) + "  summary/manifest_files.txt\n")
        return summary

    def _isaac_startup_evidence(self) -> bool:
        runtime_log = self.directory / "logs/isaac_runtime.log"
        if not runtime_log.is_file():
            return False
        text = runtime_log.read_text(errors="replace")
        return "[omni." in text or "Isaac Sim" in text or "[isaacsim." in text

    def pack(self) -> tuple[int | None, str | None]:
        try:
            with tarfile.open(self.archive, "w:gz") as archive:
                archive.add(self.directory, arcname=self.directory.name, recursive=True)
            return self.archive.stat().st_size, sha256_file(self.archive)
        except (OSError, tarfile.TarError) as exc:
            print(f"Archive failed; uncompressed directory remains: {self.directory}\n{exc}", file=sys.stderr)
            return None, None

    def run(self) -> tuple[dict, int | None, str | None]:
        old_int = signal.signal(signal.SIGINT, self._signal)
        old_term = signal.signal(signal.SIGTERM, self._signal)
        try:
            try:
                self.capture_environment()
                if not self.interrupted and not self.environment["cli"]["expected_flags_present_in_source"]:
                    self.anomalies.append("required_smoke_cli_flags_missing_in_current_source")
                elif not self.interrupted:
                    smoke_log = self.directory / "logs/smoke_stdout_stderr.log"
                    record = self._run("isaac_smoke", self.smoke_command, smoke_log,
                                       timeout=self.timeout_seconds, echo=True, env=os.environ.copy())
                    self.smoke.update({"status": record["status"], "exit_code": record["exit_code"],
                                       "started_at": record["started_at"], "ended_at": record["ended_at"],
                                       "duration_seconds": record["duration_seconds"],
                                       "command": record["command"]})
                    self.collect_runtime(self.discover_runtime(smoke_log))
                self.run_tests()
            except Exception as exc:
                self.anomalies.append(f"collector_error_{type(exc).__name__}: {exc}")
                self._stop_active()
                if self.smoke["status"] == "not_run":
                    self.smoke["status"] = "not_run"
            summary = self.finish()
            size, digest = self.pack()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
        print(f"\nSmoke status: {summary['smoke']['status']} (exit={summary['smoke']['exit_code']})")
        print(f"Diagnostic JSON found: {summary['diagnostic']['found']}")
        print("Tests: " + ", ".join(f"{name}={item['status']}" for name, item in summary["tests"].items()))
        print(f"Output directory: {self.directory}")
        print(f"Archive: {self.archive if size is not None else 'not_created'}")
        print(f"Archive size bytes: {size}")
        print(f"Archive SHA-256: {digest}")
        if size is not None:
            print("Upload only this .tar.gz archive to ChatGPT.")
        return summary, size, digest


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    root = Path(os.environ.get("ORACLE_SMOKE_OUTPUT_ROOT", Path.home() / "quadrupedarena_oracle_smoke_runs"))
    raw_timeout = os.environ.get("ORACLE_SMOKE_TIMEOUT_SECONDS", "1800")
    try:
        timeout = int(raw_timeout)
        collector = Collector(repo, root, timeout_seconds=timeout)
    except (OSError, ValueError) as exc:
        print(f"Cannot prepare collector: {exc}", file=sys.stderr)
        return 2
    summary, size, _ = collector.run()
    return 0 if size is not None and summary["smoke"]["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
