import os
import unittest

from visualization.rollout import Controller


class TestRollout(unittest.TestCase):
    def setUp(self):
        class Args(object):
            pass

        args = Args()
        args.env = "newcleanup"

        self.controller = Controller(args)

    def test_rollouts(self):
        path = os.path.abspath(os.path.dirname(__file__))
        self.controller.render_rollout(horizon=500, path=path)
        # cleanup
        if os.path.exists("tests/cleanup_trajectory.mp4"):
            os.remove("tests/cleanup_trajectory.mp4")


if __name__ == "__main__":
    unittest.main()
