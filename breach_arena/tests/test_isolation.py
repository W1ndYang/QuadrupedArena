"""Breach Arena must not depend on, or be confused with, the repository's earlier games."""
import ast
import json
import sys
import unittest
from pathlib import Path

from breach_arena import RULES_VERSION

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
ALLOWED = set(sys.stdlib_module_names) | {'numpy', 'matplotlib', 'pygame', 'breach_arena'}


class Isolation(unittest.TestCase):
    def test_imports_stay_inside_package(self):
        for f in PKG.rglob('*.py'):
            tree = ast.parse(f.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name.split('.')[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module.split('.')[0]]
                else:
                    continue
                for n in names:
                    self.assertIn(n, ALLOWED, f'{f.relative_to(REPO)} imports {n}')

    def test_no_paths_into_old_games(self):
        for f in list(PKG.rglob('*.py')) + list(PKG.rglob('*.json')):
            if f.parent.name == 'tests':
                continue
            text = f.read_text(encoding='utf-8')
            for needle in ('benchmarks/fixed_goal', 'scripts/trial', 'fixed_goal_probe', 'trial_config.json'):
                self.assertNotIn(needle, text, f'{f.relative_to(REPO)} references {needle}')

    def test_versions_are_distinct(self):
        self.assertTrue(RULES_VERSION.startswith('breach-'))
        rules = json.loads((PKG / 'rules.json').read_text(encoding='utf-8'))
        self.assertEqual(rules['version'], RULES_VERSION)
        old = REPO / 'benchmarks' / 'fixed_goal_v041' / 'RULES.md'
        if old.exists():
            self.assertNotIn(RULES_VERSION, old.read_text(encoding='utf-8'))

    def test_outputs_go_to_ignored_workspace(self):
        from breach_arena.replay import workspace
        ignore = (REPO / '.gitignore').read_text(encoding='utf-8')
        self.assertIn('.arena-workspace/', ignore)
        self.assertTrue(str(workspace()).endswith('.arena-workspace') or 'ARENA_WORKSPACE' in __import__('os').environ)


if __name__ == '__main__':
    unittest.main()
