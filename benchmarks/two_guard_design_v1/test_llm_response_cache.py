"""Saved responses must never stand in for changed prompts or paid live calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_bundle import create_bundle
from llm_compare_ubuntu import comparison_bundle, recorded_cache_from_comparison
from llm_motion_client import APIReasoner
from llm_response_cache import ExactResponseCache, add_recorded_events


CONFIG = dict(provider='deepseek', base_url='https://api.deepseek.com',
              model='deepseek-flash', thinking={'type':'disabled'},
              reasoning_effort='none', max_tokens=3072)
MESSAGES = [dict(role='system', content='Predict using observations'),
            dict(role='user', content='{"time":3.3,"observations":[[1,2]]}')]


def event():
    return dict(kind='decision', fixture_only=False, config=CONFIG,
        request=dict(request_id='one'), llm_request=MESSAGES,
        response=dict(reasoning=dict(raw_response=dict(text='{"hypotheses":[]}',
            wall_seconds=3., usage=dict(total_tokens=200), model='deepseek-flash'))))


class ResponseCacheTests(unittest.TestCase):
    def test_identical_prompt_reuses_response_without_reading_key_or_network(self):
        entries = {}
        add_recorded_events(entries, [event()], dict(source_archive_sha256='source'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cache.json'
            path.write_text(json.dumps(dict(schema='llm-exact-response-cache-v1', entries=entries)))
            model = APIReasoner(dict(CONFIG, recorded_response_cache_file=str(path),
                                     boundary_track_recovery=True))
            with patch('llm_motion_client._key', side_effect=AssertionError('No key for cache hit')):
                result = model.infer(MESSAGES)
            self.assertEqual(result['response_source'], 'recorded_exact_request')
            self.assertEqual(result['wall_seconds'], 0.)
            self.assertEqual(result['original_model_wall_seconds'], 3.)
            self.assertTrue(result['usage_is_reused'])
            changed = copy.deepcopy(MESSAGES)
            changed[-1]['content'] += ' '
            with patch('llm_motion_client._key', side_effect=FileNotFoundError('Live key required')):
                with self.assertRaisesRegex(FileNotFoundError, 'Live key required'):
                    model.infer(changed)
            self.assertIsNone(model.recorded.lookup(dict(CONFIG, max_tokens=1024), MESSAGES))

    def test_fixture_and_conflicting_outputs_cannot_silently_become_real_evidence(self):
        entries = {}
        fixture = event()
        fixture['fixture_only'] = True
        add_recorded_events(entries, [fixture], {})
        self.assertFalse(entries)
        add_recorded_events(entries, [event()], {})
        different = event()
        different['response']['reasoning']['raw_response']['text'] = '{}'
        with self.assertRaisesRegex(ValueError, 'Conflicting recorded outputs'):
            add_recorded_events(entries, [different], {})

    def test_nested_archive_import_checks_inner_bundle_and_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'run'
            run.mkdir()
            (run/'decision_events.jsonl').write_text(json.dumps(event())+'\n')
            comparison = root/'comparison'
            comparison.mkdir()
            create_bundle(run, comparison/'run.tar.gz')
            (comparison/'comparison.json').write_text('{}')
            archive = root/'comparison.tar.gz'
            comparison_bundle(comparison, archive)
            cached = recorded_cache_from_comparison(archive)
            self.assertEqual(len(cached['entries']), 1)
            entry = next(iter(cached['entries'].values()))
            self.assertEqual(entry['provenance']['source_archive_sha256'],
                             cached['source_archive_sha256'])
            self.assertEqual(entry['provenance']['request_id'], 'one')
            path = root/'cache.json'
            key = next(iter(cached['entries']))
            cached['entries'][key]['messages'][0]['content'] = 'Changed'
            path.write_text(json.dumps(cached))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                ExactResponseCache(path)


if __name__ == '__main__':
    unittest.main()
