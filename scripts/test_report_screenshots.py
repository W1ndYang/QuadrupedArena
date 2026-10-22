import unittest
from report_screenshots import video_frame_index


class ScreenshotTimingTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(fps=12,intro_seconds=2,collapse=dict(time=20,display_seconds=2))

    def test_intro_offset(self):
        self.assertEqual(video_frame_index(0,self.cfg),24)

    def test_before_collapse(self):
        self.assertEqual(video_frame_index(18,self.cfg),240)

    def test_at_collapse_uses_completed_insert(self):
        self.assertEqual(video_frame_index(20,self.cfg),288)

    def test_after_collapse(self):
        self.assertEqual(video_frame_index(22,self.cfg),312)

    def test_final_match_frame_excludes_outro(self):
        self.assertEqual(video_frame_index(40,self.cfg),528)

    def test_nonexistent_time_rejected(self):
        with self.assertRaises(ValueError):video_frame_index(.01,self.cfg)


if __name__=='__main__':unittest.main()
