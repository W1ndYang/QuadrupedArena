"""Do not hide infrastructure failures or lose evidence in paired comparisons."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from llm_bundle import create_bundle
from llm_compare_ubuntu import comparison_bundle, read_result, summarize, verify_comparison


def row(index, method, winner=None, status='completed'):
    return dict(index=index, method=method, status=status,
                summary=dict(winner=winner, seconds=10.), llm_inference_attempts=0,
                model_seconds=[], api_service_seconds=[], local_service_seconds=[])


class ComparisonTests(unittest.TestCase):
    def test_pairing_excludes_errors_explicitly_and_reports_unattempted_trials(self):
        rows = [row(0, 'llm', 'attacker'), row(0, 'numeric', 'defender'),
                row(1, 'llm', status='failed'), row(1, 'numeric', 'attacker'),
                row(2, 'llm', 'defender'), row(2, 'numeric', 'attacker')]
        result = summarize(rows, ['llm', 'numeric'], 4)
        llm = result['methods']['llm']
        self.assertEqual((llm['planned'], llm['completed'], llm['run_errors'],
                          llm['unattempted'], llm['wins']), (4, 2, 1, 1, 1))
        pairs = result['paired_vs_llm']['numeric']
        self.assertEqual((pairs['completed_pairs'], pairs['excluded_pairs'],
                          pairs['llm_only_won'], pairs['baseline_only_won']), (2, 2, 1, 1))
        self.assertEqual(pairs['exact_two_sided_p'], 1.)

    def test_observation_abstention_counts_as_paired_loss_instead_of_dropped_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'run_manifest.json').write_text(json.dumps(dict(
                episode_id='e', status='failed', replay_matches=False)))
            (root / 'e.json').write_text(json.dumps(dict(status='failed',
                error='No associated two-sample guard track after observation cap')))
            (root / 'decision_events.jsonl').write_text(json.dumps(dict(
                kind='decision', request=dict(time=24.), service_wall_seconds=.01,
                response=dict(status='error',
                    error='No associated two-sample guard track after observation cap')))+'\n')
            failure = read_result(root, 'llm', 0, 1, 2)
            self.assertEqual(failure['status'], 'policy_failure')
            self.assertFalse(failure['summary']['formal_terminal_reached'])
            result = summarize([failure, row(0, 'numeric', 'attacker')], ['llm', 'numeric'], 1)
            self.assertEqual(result['methods']['llm']['policy_failures'], 1)
            self.assertEqual(result['methods']['llm']['win_rate_on_evaluated'], 0.)
            self.assertEqual(result['paired_vs_llm']['numeric']['baseline_only_won'], 1)
            self.assertEqual(result['paired_vs_llm']['numeric']['excluded_pairs'], 0)

    def test_exact_p_for_six_discordant_wins(self):
        rows = [row(i, method, 'attacker' if method == 'llm' else 'defender')
                for i in range(6) for method in ('numeric', 'llm')]
        result = summarize(rows, ['llm', 'numeric'], 6)
        self.assertEqual(result['paired_vs_llm']['numeric']['exact_two_sided_p'], .03125)

    def test_one_archive_preserves_and_checks_individual_run_bundles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / 'runs'
            runs.mkdir()
            single = runs / 'single'
            single.mkdir()
            (single / 'run_manifest.json').write_text('{"status":"failed"}')
            create_bundle(single, runs / 'single.tar.gz')
            (runs / 'comparison.json').write_text('{"status":"failed"}')
            output = root / 'comparison.tar.gz'
            comparison_bundle(runs, output)
            manifest = verify_comparison(output)
            self.assertEqual({r['path'] for r in manifest['files']},
                             {'comparison.json', 'single.tar.gz'})
            # A valid tar container with changed bytes must fail content verification.
            with tarfile.open(output) as source:
                members = [(m.name, source.extractfile(m).read()) for m in source]
            corrupt = root / 'corrupt.tar.gz'
            with tarfile.open(corrupt, 'w:gz') as target:
                for name, data in members:
                    if name.endswith('/comparison.json'):
                        data += b' '
                    entry = tarfile.TarInfo(name)
                    entry.size = len(data)
                    target.addfile(entry, io.BytesIO(data))
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                verify_comparison(corrupt)


if __name__ == '__main__':
    unittest.main()
