"""Fresh-clone checks; no credentials, external assets, GPU or API needed."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from arena_paths import PROJECT, workspace_root, dependency_path, credential_path, rules_reference_path
from prepare_assets import install_archive


class PortabilityTests(unittest.TestCase):
    def test_text_draws_without_system_fonts(self):
        from PIL import Image, ImageDraw
        from arena_fonts import load_font
        with patch('arena_fonts.font_candidates', return_value=[]), patch.dict(os.environ, {}, clear=True):
            font = load_font(18)
            ImageDraw.Draw(Image.new('RGB', (256, 64))).text((0, 0), '机器人 baseline', font=font)

    def test_map_generation_without_system_fonts(self):
        from adaptive_motion import load_config
        from llm_trial import Match, map_image
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(PROJECT / 'configs/trial_rotating.json')
            with patch('arena_fonts.font_candidates', return_value=[]), patch.dict(os.environ, {}, clear=True):
                path = Path(tmp) / 'map.png'
                map_image(Match(cfg).observe(), path)
                self.assertGreater(path.stat().st_size, 100)

    def test_default_output_is_inside_ignored_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(workspace_root(), PROJECT / '.arena-workspace')

    def test_custom_workspace_and_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'ARENA_WORKSPACE': tmp, 'UNITREE_ROS_DIR': tmp + '/models'}):
                self.assertEqual(workspace_root(), Path(tmp).resolve())
                self.assertEqual(dependency_path('unitree_ros'), Path(tmp) / 'models')

    def test_credentials_follow_xdg_not_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'XDG_CONFIG_HOME': tmp}):
                self.assertEqual(credential_path('test.key'), Path(tmp) / 'quadruped-arena/test.key')

    def test_frozen_rules_self_contained(self):
        cfg = json.loads(rules_reference_path().read_text())
        self.assertNotIn('base_config', cfg)
        self.assertEqual(cfg['target_rotation']['lifetime_seconds'], 5)

    def test_launcher_help_without_isaac_or_acceptance(self):
        env = dict(os.environ, ARENA_PYTHON=sys.executable)
        env.pop('OMNI_KIT_ACCEPT_EULA', None)
        result = subprocess.run(['bash', str(PROJECT / 'run.sh'), '--help'],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--vision-trial', result.stdout)

    def test_explicit_eula_required(self):
        env = dict(os.environ, ARENA_PYTHON=sys.executable)
        env.pop('OMNI_KIT_ACCEPT_EULA', None)
        result = subprocess.run(['bash', str(PROJECT / 'run.sh'), '--headless'],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('OMNI_KIT_ACCEPT_EULA', result.stderr)

    def archive(self):
        data = b'fixture: not an image'
        asset = dict(id='Example', files=[dict(name='Color.jpg', size_bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest())])
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w') as z:
            z.writestr('Color.jpg', data)
            z.writestr('../unexpected.txt', b'must never be extracted')
        return asset, out.getvalue(), data

    def test_assets_only_extract_whitelisted_verified_members(self):
        asset, blob, data = self.archive()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'assets'
            install_archive(asset, blob, root)
            self.assertEqual((root / 'Example/Color.jpg').read_bytes(), data)
            self.assertEqual(len(list(Path(tmp).rglob('*'))), 3)
            install_archive(asset, blob, root)  # Existing identical files are safe.

    def test_assets_reject_hash_mismatch_before_writing(self):
        asset, blob, _ = self.archive()
        asset['files'][0]['sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                install_archive(asset, blob, Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_assets_preserve_modified_files(self):
        asset, blob, _ = self.archive()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'Example').mkdir()
            target = root / 'Example/Color.jpg'
            target.write_bytes(b'user modification')
            with self.assertRaises(FileExistsError):
                install_archive(asset, blob, root)
            self.assertEqual(target.read_bytes(), b'user modification')


if __name__ == '__main__':
    unittest.main()
