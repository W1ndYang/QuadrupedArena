"""Mac/CPU integration tests of the one-command collector's failure paths."""

import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch

from collect_oracle_v1_1_runtime_smoke import Collector, TEST_COMMANDS, safe_remote, sha256_file


REPO = Path(__file__).resolve().parents[1]
BOUNDARY = {
    "simulation_seconds": 0.0, "frame_count": 1, "render_sync_quality": "blocked",
    "full": {"status": "valid", "reason": None, "state_id": "f1", "decision_step": 0,
             "map_phase": "pre_collapse", "A.region_id": "open_traversable",
             "B.region_id": "open_traversable"},
    "limited": {"status": "valid", "reason": None, "state_id": "l1", "decision_step": 0,
                "map_phase": "pre_collapse", "A.region_id": "open_traversable"},
}


class CollectorTests(unittest.TestCase):
    def setUp(self):
        runtime_environment = patch.dict(os.environ, {"ARENA_PYTHON": sys.executable})
        runtime_environment.start()
        self.addCleanup(runtime_environment.stop)

    def test_default_collection_runs_transition_contract_tests(self):
        commands = dict(TEST_COMMANDS)
        self.assertEqual(commands["transition_trace"],
                         ("-m", "unittest", "scripts.test_oracle_transition_trace", "-v"))

    def make_collector(self, root: Path, *, diagnostic=True, diagnostic_payload=None, trace=True,
                       trace_payload=None, sleep=0, smoke_exit=0,
                       first_test_exit=0, timeout=10):
        runtime_parent = root / "runtime_parent"
        runtime = runtime_parent / "new_run"
        payload = json.dumps(diagnostic_payload if diagnostic_payload is not None else
                             {"render_sync_quality": "blocked", "same_frame_verified": False,
                              "boundaries": [BOUNDARY]})
        trace_text = json.dumps(trace_payload if trace_payload is not None else {
            "trace_version": "oracle_transition_trace_v1", "render_sync_quality": "blocked",
            "same_frame_verified": False,
            "summary": {"complete_transition_count": 1, "unclosed_action_count": 0,
                        "mapped_action_count": 1, "unmapped_action_count": 0,
                        "time_alignment_valid": True, "missing_or_quality_reasons": []},
            "transitions": [{"transition_index": 0, "render_sync_quality": "blocked",
                             "same_frame_verified": False, "terminal": False,
                             "start": {"observation_id": 0, "decision_step": 0,
                                       "simulation_seconds": 0., "limited": {}},
                             "next": {"observation_id": 1, "decision_step": 1,
                                      "simulation_seconds": 2., "limited": {}},
                             "actual_elapsed_seconds": 2.,
                             "time_alignment": {"valid": True, "errors": [], "actual_elapsed_seconds": 2.},
                             "action": {"action_mapping_status": "mapped", "oracle_action_id": "mock"}}],
            "unclosed_actions": []})
        code = ("from pathlib import Path; import sys,time; "
                "p=Path(sys.argv[1]);p.mkdir(parents=True); "
                "print('Run: '+str(p),flush=True); "
                "(p/'runtime.log').write_text('mock runtime\\n'); "
                "(p/'result.json').write_text('{\"status\":\"passed\"}'); "
                "(p/'frames').mkdir();(p/'frames'/'image.png').write_bytes(b'fake image'); "
                + (f"(p/'oracle_state_diagnostic.json').write_text({payload!r}); " if diagnostic else "")
                + (f"(p/'oracle_transition_trace.json').write_text({trace_text!r}); " if trace else "")
                + f"time.sleep({sleep}); sys.exit({smoke_exit})")
        tests = (("oracle", ("-c", f"raise SystemExit({first_test_exit})")),
                 ("related", ("-c", "print('related passed')")),
                 ("diff_check", ("git", "diff", "--check")))
        return Collector(REPO, root / "output", smoke_command=(sys.executable, "-c", code, str(runtime)),
                         test_commands=tests, timeout_seconds=timeout,
                         runtime_parent=runtime_parent)

    def test_success_manifest_and_secret_redaction(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"FAKE_API_SECRET": "SECRET_SENTINEL_123"}):
            root = Path(temp)
            stale = root / "runtime_parent" / "old_run"
            stale.mkdir(parents=True)
            (stale / "oracle_state_diagnostic.json").write_text("STALE_DIAGNOSTIC")
            collector = self.make_collector(root)
            summary, size, digest = collector.run()
            self.assertEqual(summary["smoke"]["status"], "success")
            self.assertTrue(summary["diagnostic"]["found"])
            self.assertEqual(summary["diagnostic"]["boundary_count"], 1)
            self.assertEqual(summary["diagnostic"]["render_sync_quality"], "blocked")
            self.assertFalse(summary["diagnostic"]["same_frame_verified"])
            self.assertTrue(summary["transition_trace"]["found"])
            self.assertEqual(summary["transition_trace"]["complete_transition_count"], 1)
            self.assertTrue(summary["transition_trace"]["time_alignment_valid"])
            self.assertEqual(summary["tests"]["related"]["status"], "success")
            self.assertGreater(size, 0)
            self.assertEqual(len(digest), 64)
            manifest = collector.directory / "summary/manifest_files.txt"
            for line in manifest.read_text().splitlines()[1:]:
                relative, byte_count, checksum = line.split("\t")
                self.assertEqual((collector.directory / relative).stat().st_size, int(byte_count))
                self.assertEqual(sha256_file(collector.directory / relative), checksum)
            recorded_manifest_hash = (collector.directory / "summary/manifest.sha256").read_text().split()[0]
            self.assertEqual(sha256_file(manifest), recorded_manifest_hash)
            with tarfile.open(collector.archive, "r:gz") as archive:
                members = archive.getnames()
                self.assertTrue(any(name.endswith("/summary/manifest_files.txt") for name in members))
                self.assertTrue(any(name.endswith("/diagnostics/oracle_state_diagnostic.json") for name in members))
                self.assertTrue(any(name.endswith("/diagnostics/oracle_transition_trace.json") for name in members))
                self.assertFalse(any(name.endswith("/frames/image.png") for name in members))
                self.assertFalse(any("old_run" in name for name in members))
            bundle_text = "\n".join(p.read_text(errors="replace") for p in collector.directory.rglob("*") if p.is_file())
            self.assertNotIn("SECRET_SENTINEL_123", bundle_text)
            self.assertNotIn("STALE_DIAGNOSTIC", bundle_text)

    def test_failed_smoke_missing_diagnostic_and_test_failure_still_pack(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = self.make_collector(Path(temp), diagnostic=False, smoke_exit=7, first_test_exit=9)
            summary, size, _ = collector.run()
            self.assertEqual(summary["smoke"]["status"], "failed")
            self.assertEqual(summary["smoke"]["exit_code"], 7)
            self.assertFalse(summary["diagnostic"]["found"])
            self.assertIn("diagnostic_json_missing", summary["anomalies"])
            self.assertEqual(summary["tests"]["oracle"]["status"], "failed")
            self.assertEqual(summary["tests"]["related"]["status"], "success")
            self.assertEqual(summary["tests"]["diff_check"]["status"], "success")
            self.assertTrue(collector.archive.exists())
            self.assertGreater(size, 0)

    def test_missing_trace_is_reported_and_failure_archive_is_still_built(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = self.make_collector(Path(temp), trace=False)
            summary, size, _ = collector.run()
            self.assertEqual(summary["smoke"]["status"], "failed")
            self.assertFalse(summary["transition_trace"]["found"])
            self.assertIn("oracle_transition_trace_missing", summary["anomalies"])
            self.assertTrue(collector.archive.exists())
            self.assertGreater(size, 0)

    def test_invalid_trace_alignment_is_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            bad = {"trace_version": "oracle_transition_trace_v1", "render_sync_quality": "blocked",
                   "same_frame_verified": False,
                   "summary": {"complete_transition_count": 1, "unclosed_action_count": 0,
                               "mapped_action_count": 1, "unmapped_action_count": 0,
                               "time_alignment_valid": True},
                   "transitions": [{"transition_index": 0, "render_sync_quality": "blocked",
                                    "same_frame_verified": False, "terminal": False,
                                    "start": {"observation_id": 0, "decision_step": 0,
                                              "simulation_seconds": 0., "limited": {}},
                                    "next": {"observation_id": 2, "decision_step": 2,
                                             "simulation_seconds": 2., "limited": {}},
                                    "time_alignment": {"valid": True, "errors": [], "actual_elapsed_seconds": 2.},
                                    "actual_elapsed_seconds": 2.,
                                    "action": {"action_mapping_status": "mapped", "oracle_action_id": "mock"}}],
                   "unclosed_actions": []}
            collector = self.make_collector(Path(temp), trace_payload=bad)
            summary, size, _ = collector.run()
            self.assertEqual(summary["smoke"]["status"], "failed")
            self.assertTrue(summary["transition_trace"]["found"])
            self.assertIsNotNone(summary["transition_trace"]["parse_error"])
            self.assertTrue((collector.directory / "diagnostics/oracle_transition_trace.json").exists())
            self.assertGreater(size, 0)

    def test_runtime_python_is_used_and_recorded_for_both_test_groups(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"ARENA_PYTHON": sys.executable}):
            collector = self.make_collector(Path(temp))
            summary, _, _ = collector.run()
            expected = str(Path(sys.executable).absolute())
            self.assertEqual(summary["smoke"]["status"], "success")
            for name in ("oracle", "related"):
                record = summary["tests"][name]
                self.assertEqual(record["status"], "success")
                self.assertEqual(record["interpreter"], expected)
                self.assertEqual(record["interpreter_probe"]["status"], "success")
                log = (collector.directory / record["log"]).read_text()
                self.assertIn(expected + " -c", log)
                self.assertIn("Runtime Python --version: Python ", log)
                self.assertIn('NumPy import/version: {"status": "success", "version":', log)
            self.assertEqual(summary["tests"]["diff_check"]["status"], "success")
            self.assertIn("Command: git diff --check", (collector.directory / "tests/diff_check.log").read_text())

    def test_missing_or_nonexecutable_runtime_python_blocks_tests_without_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            not_executable = root / "not_executable_python"
            not_executable.write_text("#!/bin/sh\nexit 0\n")
            for index, configured in enumerate((str(root / "missing_python"), str(not_executable))):
                with self.subTest(configured=configured), patch.dict(os.environ, {"ARENA_PYTHON": configured}):
                    collector = self.make_collector(root / f"case_{index}")
                    summary, size, _ = collector.run()
                    self.assertEqual(summary["smoke"]["status"], "success")
                    self.assertGreater(size, 0)
                    self.assertEqual(summary["tests"]["oracle"]["status"], "blocked")
                    self.assertEqual(summary["tests"]["related"]["status"], "blocked")
                    self.assertEqual(summary["tests"]["diff_check"]["status"], "success")
                    self.assertIsNone(summary["tests"]["oracle"]["interpreter"])
                    self.assertIn("No system Python fallback", (collector.directory / "tests/oracle.log").read_text())
                    self.assertIsNone(summary["dependency_probe"]["probe_python_command"])

    def test_missing_numpy_blocks_python_tests_and_keeps_smoke_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_python = root / "python_without_numpy"
            fake_python.write_text("#!/bin/sh\n"
                                   "if [ \"$1\" = --version ]; then echo 'Python 3.11.0'; else "
                                   "echo '{\"numpy_import\":{\"status\":\"failed\",\"error\":\"ModuleNotFoundError: numpy\"}}'; fi\n")
            fake_python.chmod(0o755)
            with patch.dict(os.environ, {"ARENA_PYTHON": str(fake_python)}):
                collector = self.make_collector(root)
                summary, _, _ = collector.run()
            self.assertEqual(summary["smoke"]["status"], "success")
            self.assertEqual(summary["tests"]["oracle"]["status"], "blocked")
            self.assertEqual(summary["tests"]["related"]["status"], "blocked")
            self.assertEqual(summary["tests"]["diff_check"]["status"], "success")
            self.assertEqual(summary["dependency_probe"]["numpy_import"]["status"], "failed")
            self.assertIn("ModuleNotFoundError: numpy", (collector.directory / "tests/related.log").read_text())

    def test_timeout_terminates_and_packs(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = self.make_collector(Path(temp), diagnostic=False, sleep=5, timeout=1)
            summary, size, _ = collector.run()
            self.assertEqual(summary["smoke"]["status"], "timeout")
            self.assertEqual(summary["smoke"]["exit_code"], 124)
            self.assertGreater(size, 0)

    def test_malformed_current_diagnostic_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = self.make_collector(Path(temp), diagnostic_payload={"boundaries": "wrong"})
            summary, size, _ = collector.run()
            self.assertEqual(summary["smoke"]["status"], "failed")
            self.assertTrue(summary["diagnostic"]["found"])
            self.assertIsNotNone(summary["diagnostic"]["parse_error"])
            self.assertEqual(json.loads((collector.directory / "diagnostics/oracle_state_diagnostic.json").read_text()),
                             {"boundaries": "wrong"})
            self.assertGreater(size, 0)

    def test_runtime_symlink_cannot_copy_unrelated_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            collector = self.make_collector(root, diagnostic=False)
            runtime = collector.runtime_parent / "new_run"
            runtime.mkdir(parents=True)
            secret = root / "unrelated_secret.txt"
            secret.write_text("NOT_FOR_UPLOAD")
            (runtime / "oracle_state_diagnostic.json").symlink_to(secret)
            collector.collect_runtime(runtime)
            self.assertFalse(collector.diagnostic["found"])
            self.assertFalse((collector.directory / "diagnostics/oracle_state_diagnostic.json").exists())

    def test_interruption_requests_termination_and_packs(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = self.make_collector(Path(temp), diagnostic=False, sleep=5)
            timer = threading.Timer(.5, lambda: collector._signal(signal.SIGINT, None))
            timer.start()
            try:
                summary, size, _ = collector.run()
            finally:
                timer.cancel()
                timer.join()
            self.assertEqual(summary["smoke"]["status"], "interrupted")
            self.assertEqual(summary["tests"]["oracle"]["status"], "not_run_interrupted")
            self.assertGreater(size, 0)

    def test_remote_redaction(self):
        self.assertEqual(safe_remote("https://user:TOP_SECRET@github.com/team/repo.git?token=SECRET"),
                         {"scheme": "https", "host": "github.com", "repository_basename": "repo.git"})
        self.assertEqual(safe_remote("git@github.com:team/repo.git")["host"], "github.com")


if __name__ == "__main__":
    unittest.main()
