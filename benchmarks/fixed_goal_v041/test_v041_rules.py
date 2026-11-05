"""Tests for the active goal-only wrapper, rather than the legacy tag-win clock."""
import unittest
import ds_fixed_goal  # Installs the isolated snapshot's module search path.
from run_goal_only import GoalOnlyClock

class GoalOnlyRules(unittest.TestCase):
    def locked(self,valid):
        c=GoalOnlyClock();c.step(0,valid)
        for _ in range(50):result=c.step(.02,valid)
        return c,result

    def test_attacker_tag_only_disables(self):
        c,result=self.locked([True,False])
        self.assertEqual(result,(None,None,[True,False]))
        self.assertAlmostEqual(c.disabled_until-c.time,1.)
        self.assertAlmostEqual(c.cooldown_until[0]-c.time,4.)

    def test_defender_tag_ends_match(self):
        _,result=self.locked([False,True])
        self.assertEqual(result[:2],('defender','tag'))

    def test_goal_and_defender_tag_draw(self):
        c=GoalOnlyClock();c.step(0,[False,True])
        for _ in range(49):c.step(.02,[False,True])
        self.assertEqual(c.step(.02,[False,True],goal=True)[:2],('draw','simultaneous'))

    def test_reciprocal_tags_do_not_give_attacker_a_win(self):
        _,result=self.locked([True,True])
        self.assertEqual(result[:2],('defender','tag'))

    def test_disabled_defender_and_attacker_cooldown(self):
        c,_=self.locked([True,False])
        for _ in range(40):
            self.assertEqual(c.step(.02,[True,True]),(None,None,[False,False]))
            self.assertEqual(c.age,[0.,0.])

    def test_goal_at_deadline_wins(self):
        self.assertEqual(GoalOnlyClock().step(0,[False,False],goal=True,timeout=True)[:2],('attacker','goal'))

if __name__=='__main__':unittest.main()
