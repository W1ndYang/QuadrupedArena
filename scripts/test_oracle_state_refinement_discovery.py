"""Checks for ranking new prefixes by competing observable state splits."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from finite_state_optimizer import fit_one_bit_state, geometry_predicates
from oracle_state_refinement_discovery import analyze_boundary, equivalent_predicates


class StateRefinementDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.predicates = geometry_predicates([{"name": "P1", "center": [-4.0, -1.5],
                                                "radius": .55}])
        self.training = [
            {"sample_id": "west", "base_state_id": "root", "own_xy": [-4.327, -1.143],
             "returns": {"BASE": 0.0, "SCAN": 2.62}},
            {"sample_id": "other1", "base_state_id": "root", "own_xy": [-4.022, -1.617],
             "returns": {"BASE": 4.0}},
            {"sample_id": "other2", "base_state_id": "root", "own_xy": [-4.155, -1.239],
             "returns": {"BASE": 4.0}},
        ]

    def test_equal_value_predicates_are_exposed_for_active_sampling(self):
        fit = fit_one_bit_state(self.training, self.predicates, "BASE")
        tied = equivalent_predicates(fit, self.predicates, self.training[0]["own_xy"])
        self.assertEqual([item["predicate"]["id"] for item in tied],
                         ["P1:x:-0.5r", "P1:y:+0.5r"])

    def test_disagreement_at_exact_root_gets_first_priority(self):
        fit = fit_one_bit_state(self.training, self.predicates, "BASE")
        tied = equivalent_predicates(fit, self.predicates, self.training[0]["own_xy"])
        with TemporaryDirectory() as tmp:
            raw = Path(tmp)
            boundaries = [{"limited": {"status": "valid", "state_id": "other",
                                       "fields": {}}, "score": 0.0, "hp": [3, 3],
                           "finished": None} for _ in range(4)]
            boundaries.append({"limited": {"status": "valid", "state_id": "root",
                                            "fields": {"target.active": "P2",
                                                       "A.region_id": "target_P1", "A.hp": 2}},
                               "score": 4.0, "hp": [2, 3], "finished": None})
            (raw / "boundaries.json").write_text(json.dumps(boundaries))
            (raw / "frame_index.json").write_text(json.dumps([
                {"simulation_seconds": 8.0, "own_xy": [-4.4, -1.4]}]))
            record = {"status": "passed", "raw_dir": str(raw), "task_key": "key",
                      "result": {"stop_reason": "planned_boundary"}}
            item = analyze_boundary("new_prefix", record, tied, "root")
            self.assertTrue(item["predicate_disagreement"])
            self.assertEqual(item["priority_rank"], 0)
            self.assertTrue(item["exact_prior_root_state"])


if __name__ == "__main__":
    unittest.main()
