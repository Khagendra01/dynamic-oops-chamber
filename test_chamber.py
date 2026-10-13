"""Stdlib unit tests: python -m unittest -v test_chamber.py"""
import unittest
from chamber import Cuboid, ChamberEnv, example_scene
from planners import OracleTimeExpandedAStar
from run_demo import run


class ChamberTests(unittest.TestCase):
    def test_obstacle_bounce_and_velocity(self):
        env = ChamberEnv(start=(.1,.1,.1),goal=(.9,.9,.9), obstacles=[
            Cuboid('moving',(.5,.5,.5),(.2,.2,.2),(.5,0,0))])
        o = env.obstacles[0]
        self.assertAlmostEqual(o.center_at(.8,env.chamber_size)[0], .9)
        self.assertAlmostEqual(o.center_at(1.2,env.chamber_size)[0], .7)
        self.assertLess(o.velocity_at(1.2,env.chamber_size)[0], 0)

    def test_swept_collision_even_when_endpoints_clear(self):
        env = ChamberEnv(start=(.1,.5,.5),goal=(.9,.5,.5), obstacles=[
            Cuboid('middle',(.5,.5,.5),(.2,.2,.2))], dt=1, max_speed=1)
        self.assertTrue(env.is_state_free((.1,.5,.5),0))
        self.assertTrue(env.is_state_free((.9,.5,.5),1))
        self.assertFalse(env.is_motion_safe((.1,.5,.5),(.9,.5,.5),0,1))
        result = env.step((.8,0,0))
        self.assertTrue(result.safety_intervention)
        self.assertFalse(result.collision)
        self.assertEqual(result.observation.position,(.1,.5,.5))

    def test_moving_obstacle_hits_stationary_agent_between_samples(self):
        env = ChamberEnv(start=(.5,.5,.5),goal=(.9,.9,.9), obstacles=[
            Cuboid('cross',(.2,.5,.5),(.1,.1,.1),(.6,0,0))], dt=1.0)
        self.assertFalse(env.is_motion_safe(env.position,env.position,0,1))
        result = env.step((0,0,0))
        self.assertTrue(result.collision)
        self.assertTrue(result.done)

    def test_no_start_overlap(self):
        with self.assertRaises(ValueError):
            ChamberEnv(obstacles=[Cuboid('bad',(.1,.1,.1),(.1,.1,.1))])

    def test_reproducible_random_scene(self):
        a = example_scene('random',42)
        b = example_scene('random',42)
        self.assertEqual(a.obstacles,b.obstacles)
        self.assertEqual(a.observe(),b.observe())
        result_a = a.step((0,0,0))
        result_b = b.step((0,0,0))
        self.assertEqual(result_a,result_b)

    def test_wait_may_be_safe_when_path_is_not(self):
        env = example_scene('static')
        self.assertTrue(env.is_motion_safe(env.position, env.position,0,.5))
        self.assertFalse(env.is_motion_safe((.1,.5,.5),(.9,.5,.5),0,1))

    def test_empty_baseline_reaches_goal(self):
        summary,_ = run('empty',verbose=False)
        self.assertTrue(summary['success'])
        self.assertFalse(summary['collision'])

    def test_static_baseline_reaches_goal(self):
        summary,_ = run('static',verbose=False)
        self.assertTrue(summary['success'])
        self.assertFalse(summary['collision'])

    def test_moving_obstacles_deterministic_replay(self):
        summary_a,records_a = run('moving',7,verbose=False)
        summary_b,records_b = run('moving',7,verbose=False)
        summary_a.pop('wall_clock_s');summary_b.pop('wall_clock_s')
        self.assertEqual(summary_a,summary_b)
        self.assertEqual(records_a,records_b)


if __name__ == '__main__':
    unittest.main()
