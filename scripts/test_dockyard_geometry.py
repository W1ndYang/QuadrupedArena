"""CPU map-design regression checks; does not invoke Isaac or model APIs."""
import hashlib
import math
from pathlib import Path
import unittest
import numpy as np
from dockyard_geometry import load_design,solids,occupied,ramp_height,validate_design,ground_route


class DockyardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=load_design('dockyard_v1.json');cls.before=solids(cls.d);cls.after=solids(cls.d,True)

    def test_review_only_and_footprint(self):
        self.assertFalse(self.d['match_integration_ready'])
        self.assertEqual(self.d['stage'],'map_review_only')
        self.assertEqual(np.prod(self.d['playable_size_m']),768)
        self.assertEqual(len(self.d['walls']),14)

    def test_tunnels_are_panels_not_filled_boxes(self):
        for c in self.d['containers']:
            panels=[s for s in self.before if s['name'].startswith(c['name']+'_')]
            self.assertEqual(len(panels),6)
            if c['open_ends']:
                x,y=c['center']
                for along in np.linspace(-c['length']/2-.2,c['length']/2+.2,13):
                    for across in [-.8,0,.8]:
                        point=[x+along,y+across,1.4] if c['axis']=='x' else [x+across,y+along,1.4]
                        self.assertIsNone(occupied(point,self.before))
            else:self.assertTrue(any('end' in s['name'] for s in panels))

    def test_bridge_has_ground_and_upper_levels(self):
        self.assertIsNone(occupied([0,0,1.4],self.before))
        self.assertEqual(occupied([0,0,2.1],self.before),'BridgeDeck')
        self.assertIsNone(occupied([0,0,2.6],self.before))
        for ramp in [s for s in self.before if s['shape']=='ramp']:
            self.assertAlmostEqual(ramp_height(ramp,ramp['high_x']),2.2)
            self.assertAlmostEqual(ramp_height(ramp,ramp['low_x']),.025)

    def test_collapse_only_changes_one_wall(self):
        changes=[a['name'] for a,b in zip(self.before,self.after) if a!=b]
        self.assertEqual(changes,['High3'])
        wall=next(s for s in self.after if s['name']=='High3')
        self.assertEqual(wall['center'],[-.4,3.9,.2]);self.assertEqual(wall['size'],[2.4,2.2,.4])
        self.assertEqual(self.d['collapse']['time'],20)

    def test_connectivity_and_clearance(self):
        v=validate_design(self.d)
        self.assertEqual(v['status'],'passed')
        self.assertGreater(v['checks'],300)
        self.assertLess(v['ramp_slope_degrees'],20)
        self.assertEqual(len(v['paths']),6)
        self.assertGreater(v['paths']['before_P1_to_P2']['length_m'],15)

    def test_blocked_start_rejected(self):
        self.assertIsNone(ground_route(self.d,self.before,[-3.15,-3.6],[0,-2]))

    def test_archived_rules_not_edited(self):
        from arena_paths import rules_reference_path
        p=rules_reference_path()
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'806b467e5cd019af6e50896d000be26212494f399f2713de1420ef6a72453a8a')


if __name__=='__main__':unittest.main()
