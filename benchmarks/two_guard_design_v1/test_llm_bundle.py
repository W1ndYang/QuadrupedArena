"""One-file handoff must preserve logs and exclude credentials."""
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from llm_bundle import create_bundle, verify_bundle


class LLMBundleTests(unittest.TestCase):
    def test_bundle_contains_logs_episode_and_verified_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run'
            run.mkdir()
            (run / 'decision_events.jsonl').write_text('{"kind":"decision"}\n')
            (run / 'episode.json').write_text(json.dumps({
                'episode_id': 'e', 'status': 'completed', 'summary': {'winner': 'defender'}}))
            (run / 'config_effective.json').write_text('{"api_key_env":"DEEPSEEK_API_KEY"}')
            (run / 'unrelated.txt').write_text('do not include')
            archive = root / 'run.tar.gz'
            result = create_bundle(run, archive, ['private-key-value'])
            manifest = verify_bundle(archive)
            self.assertEqual(result['file_count'], 3)
            self.assertEqual({row['path'] for row in manifest['files']},
                             {'decision_events.jsonl', 'episode.json',
                              'config_effective.json'})
            with tarfile.open(archive, 'r:gz') as bundle:
                self.assertEqual(bundle.extractfile('llm_run/episode.json').read(),
                                 (run / 'episode.json').read_bytes())

    def test_credentials_block_archive_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run'
            run.mkdir()
            (run / 'service_stdout.log').write_text('Error private-key-value')
            archive = root / 'run.tar.gz'
            with self.assertRaisesRegex(ValueError, 'Credential value found'):
                create_bundle(run, archive, ['private-key-value'])
            self.assertFalse(archive.exists())


if __name__ == '__main__':
    unittest.main()
