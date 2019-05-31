from unittest import TestCase
from agents.Unity_Env import UnityEnv_Reacher
import torch
from copy import copy
import numpy as np

# class UnityEnv_ReacherTest1(TestCase):

#     def setUp(self):
#         self.start = True

        
#     def test_env1size(self):
#         self.env_1 = UnityEnv_Reacher(env_file="Reacher_Linux/Reacher.x86_64")
#         self.state_size1 = copy(self.env_1.state_size)
#         self.action_size1 = copy(self.env_1.action_size)
#         self.num_agents1 = copy(self.env_1.num_agents)
#         self.states1 = copy(self.env_1.reset())
#         self.env_1.close()
#         self.assertEqual(self.state_size1, 33)
#         self.assertEqual(self.action_size1, 4)
#         self.assertEqual(self.num_agents1, 1)
#         self.assertTrue(self.states1.shape == (1, self.state_size1))

class UnityEnv_ReacherTest2(TestCase):

    def setUp(self):
        self.start = True
        self.env_2 = UnityEnv_Reacher(env_file="Reacher_Linux_20/Reacher.x86_64")

    def test_env2size(self):
        self.state_size2 = copy(self.env_2.state_size)
        self.action_size2 = copy(self.env_2.action_size)
        self.num_agents2 = copy(self.env_2.num_agents)
        self.states2 = copy(self.env_2.reset())
        self.assertEqual(self.state_size2, 33)
        self.assertEqual(self.action_size2, 4)
        self.assertEqual(self.num_agents2, 20)
        self.assertTrue(self.states2.shape == (20, self.state_size2))
        actions = np.array([np.array([0.2, 0.3, 0.4, 0.5]) for _ in range(20)])
        next_states, rewards, dones, _ = self.env_2.step(actions)
        self.assertTrue(next_states.shape == (20, self.state_size2))
        self.assertTrue((rewards.shape == (20, )))

        ## TODO add test for states ranges after running 1 episodes
