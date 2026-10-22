"""CPU geometry/asset tests; native USD assertions run inside art_scene.py."""
import json
from pathlib import Path
import unittest
import numpy as np
from arena_art import beveled_box, asset_manifest


class ArtTests(unittest.TestCase):
    def test_cover_bounds_and_outward_faces(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/trial_v3.json').read_text())
        for cover in cfg['arena']['covers']:
            with self.subTest(cover=cover['name']):
                points,faces=beveled_box(cover['size'],.025)
                np.testing.assert_allclose(np.ptp(points,axis=0),cover['size'])
                np.testing.assert_allclose(points.mean(axis=0),0,atol=1e-15)
                triangles=points[faces]
                normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
                self.assertTrue(np.all(np.sum(normals*triangles.mean(axis=1),axis=1)>0))

    def test_thin_parts_have_nonzero_volume(self):
        p,f=beveled_box([.04,.02,.003],.02)
        self.assertTrue(np.isfinite(p).all())
        self.assertGreater(len(f),10)
        np.testing.assert_allclose(np.ptp(p,axis=0),[.04,.02,.003])

    def test_playable_dimensions_unchanged(self):
        root=Path(__file__).resolve().parents[1]/'configs'
        art=json.loads((root/'art_v1.json').read_text())
        rules=json.loads((root/art['rules_config']).read_text())
        self.assertEqual(art['playable_size_m'],rules['arena']['floor_size'][:2])
        self.assertTrue(all(a>b for a,b in zip(art['compound_size_m'],art['playable_size_m'])))

    def test_local_pbr_inputs(self):
        from arena_paths import workspace_root
        workspace=workspace_root()
        if not (workspace/'simulators/assets/quadruped-arena/pbr/Concrete023').is_dir():
            self.skipTest('Optional PBR integration test: run scripts/prepare_assets.py first')
        assets=asset_manifest(workspace/'simulators/assets/quadruped-arena/pbr')
        self.assertEqual(len(assets),3)
        self.assertTrue(all(a['license']=='CC0-1.0' and len(a['files'])==3 for a in assets))


if __name__=='__main__':unittest.main()
