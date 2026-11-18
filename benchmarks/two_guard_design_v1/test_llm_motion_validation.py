"""Invalid alternatives must increase unknown mass, not erase it or stop a batch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from llm_decision_service import ServiceState
from llm_motion_client import FixtureReasoner
from llm_semantic import build_map,COMPACT_SCHEMA_VERSION,normalize_validated_motion
from test_llm_decision import public_request,sample


def response():
    m=build_map()
    def hypothesis(name,p,edges,state='MOVE'):
        return dict(id=name,probability=p,state=state,edge_ids=edges,dwell_node_ids=[],
            speed_range_mps=[.8,1.5],dwell_range_seconds=[0.,2.],reason='Test alternative')
    return dict(schema=COMPACT_SCHEMA_VERSION,map_id=m['map_id'],track_id='track-0001',
        hypotheses=[hypothesis('valid',.7,['first_x1_y3__first_x1_y2']),
                    hypothesis('broken',.2,['first_x1_y3__first_x1_y2',
                                             'first_x1_y1__first_x1_y0']),
                    hypothesis('unknown',.1,[],'UNKNOWN')])


class MotionValidationTests(unittest.TestCase):
    def test_invalid_route_mass_is_unknown_and_valid_route_is_unchanged(self):
        raw=response();before=copy.deepcopy(raw)
        parsed,report=normalize_validated_motion(raw,build_map(),'track-0001','first',8.,'unknown_mass')
        self.assertEqual(raw,before)
        self.assertEqual(parsed['hypotheses'][0]['probability'],.7)
        self.assertEqual(parsed['hypotheses'][0]['edge_ids'],raw['hypotheses'][0]['edge_ids'])
        self.assertAlmostEqual(sum(h['probability'] for h in parsed['hypotheses'] if h['state']=='UNKNOWN'),.3)
        self.assertAlmostEqual(sum(h['probability'] for h in parsed['hypotheses']),1.)
        self.assertEqual(report['rejected_hypotheses'][0]['id'],'broken')
        with self.assertRaisesRegex(ValueError,'Disconnected'):
            normalize_validated_motion(raw,build_map(),'track-0001','first',8.)

    def test_global_identity_and_probability_errors_cannot_be_salvaged(self):
        for field,value in (('track_id','wrong'),('map_id','wrong')):
            raw=response();raw[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'mismatch'):
                normalize_validated_motion(raw,build_map(),'track-0001','first',8.,'unknown_mass')
        raw=response();raw['hypotheses'][0]['probability']=.3
        with self.assertRaisesRegex(ValueError,'sum to 1'):
            normalize_validated_motion(raw,build_map(),'track-0001','first',8.,'unknown_mass')

    def test_all_unknown_reobserves_then_reports_model_failure_at_cap(self):
        raw=response();raw['hypotheses'][0]['edge_ids']=raw['hypotheses'][1]['edge_ids']
        config=dict(provider='deepseek',invalid_hypothesis_mode='unknown_mass',max_peeks_per_stage=2)
        with tempfile.TemporaryDirectory() as directory:
            for peek in (1,2):
                state=ServiceState(config,Path(directory)/str(peek),fixture=True)
                state.orchestrator.reasoner=FixtureReasoner(lambda _:raw)
                result=state.decide(public_request(time=3.3,peek=peek,observations=[
                    sample(3.1,[-3.,3.]),sample(3.2,[-3.,2.9])]))
                if peek==1:
                    self.assertEqual(result['status'],'observe')
                    self.assertAlmostEqual(result['reasoning']['validation']['rejected_probability_mass'],.9)
                else:
                    self.assertEqual(result['error_type'],'ModelOutputError')

    def test_bad_json_is_model_failure_but_public_protocol_errors_stay_errors(self):
        class BadJSON:
            def infer(self,messages):return dict(text='not json',wall_seconds=2.)
        with tempfile.TemporaryDirectory() as directory:
            state=ServiceState(dict(provider='deepseek',invalid_hypothesis_mode='unknown_mass'),directory,fixture=True)
            state.orchestrator.reasoner=BadJSON()
            request=public_request(time=3.3,peek=2,observations=[
                sample(3.1,[-3.,3.]),sample(3.2,[-3.,2.9])])
            self.assertEqual(state.decide(request)['error_type'],'ModelOutputError')
            request=copy.deepcopy(request);request['request_id']='future'
            request['observations'][-1]['time']=4.
            self.assertEqual(state.decide(request)['error_type'],'ValueError')


if __name__=='__main__':unittest.main()
