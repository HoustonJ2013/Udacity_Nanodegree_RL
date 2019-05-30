from unittest import TestCase
import numpy as np
from agents.ddpg_agent import Agent, OUNoise
import torch


class AgentTest(TestCase):

    def setUp(self):
        self.state_size = 10
        self.action_size = 4
        self.batch_size = 64
        seed = 40
        self.agent = Agent(state_size=self.state_size, 
                           action_size=self.action_size, 
                           batch_size=self.batch_size,
                           random_seed=seed, 
                           device=torch.device("cpu"))

    def test_act(self):
        state = np.random.random((100, self.state_size))
        act = self.agent.act(state, add_noise=True)
        self.assertTrue(act.dtype, np.float32)
        self.assertEqual(act.shape, (100, self.action_size))



class OUNoiseTest(TestCase):

    def setUp(self):
        self.size = (64,)
        self.mu = 0.5
        seed = 40
        self.ou_noise = OUNoise(size=self.size, seed=seed, mu=self.mu)

    def test_reset(self):
        self.ou_noise.reset()
        self.assertEqual(self.ou_noise.state.shape, self.size)
        self.assertTrue((self.ou_noise.state == self.mu * np.ones(self.size)).all())
    
    def test_sample(self):
        sample = self.ou_noise.sample()
        self.assertEqual(sample.shape, self.size)
