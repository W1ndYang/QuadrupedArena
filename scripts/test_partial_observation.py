import unittest
from partial_observation import observation,validate_action


class PartialObservationTests(unittest.TestCase):
    def packet(self,index=1,secret=None):
        own=dict(position=[1,2,.4],yaw=0,linear_velocity=[0,0,0],hp=3,enemy_xy=secret)
        public=dict(score=1,active_target=0,human_memory=secret,human_route=secret,aim=secret,geometry=secret)
        return observation(index,2.,own,public,'001.png',[5,2.8])

    def test_hidden_state_cannot_change_packet(self):
        self.assertEqual(self.packet(secret=[1,2]),self.packet(secret=[-5,-3]))

    def test_initial_position_only_at_zero(self):
        self.assertIn('opponent_initial_position',self.packet(index=0))
        self.assertNotIn('opponent_initial_position',self.packet())

    def test_no_truth_tracking_action(self):
        for a in ({'skill':'AIM'},{'skill':'HOLD','facing':'enemy'}, {'skill':'MOVE','goal':[1,2],'enemy_xy':[4,5]}, {'skill':'TURN','yaw':float('nan')}):
            with self.assertRaises(ValueError):validate_action(a)

    def test_valid_self_map_actions(self):
        for a in ({'skill':'MOVE','goal':[-4,-1.5]},{'skill':'TURN','yaw':1.2},{'skill':'STOP'},{'skill':'HOLD','attack':True}):
            self.assertEqual(validate_action(a),a)


if __name__=='__main__':unittest.main()
