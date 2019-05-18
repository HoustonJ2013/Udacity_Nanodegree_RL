from unittest import TestCase
import numpy as np
from dqn.dqn_agent import Agent, ReplayBuffer
import torch


class ReplayBufferTest(TestCase):

    def setUp(self):
        self.action_size = 1
        self.buffer_size = 1000
        self.batch_size = 64
        seed=42
        self.replaybuffer = ReplayBuffer( 
                                 action_size=self.action_size, 
                                 buffer_size=self.buffer_size,
                                 batch_size=self.batch_size,
                                 seed=seed, 
                                 device=torch.device("cpu"))

    def test_add(self):
        state, action, reward, next_state, done = \
            np.array([3, 4, 1]), np.array([2]), 3, np.array([2, 3, 5]), False
        for i_ in range(2 * self.buffer_size):
            self.replaybuffer.add(state, action, reward, next_state, done)
            if i_ < self.buffer_size:
                self.assertEqual(len(self.replaybuffer), i_ + 1)
            else:
                self.assertEqual(len(self.replaybuffer), self.buffer_size)
    
    def test_sample(self):
        
        for i_ in range(2 * self.buffer_size):
            state, action, reward, next_state, done = \
            np.array([3, 4, i_]), np.array([2]), 3, np.array([2, 3, 5]), False
            self.replaybuffer.add(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = self.replaybuffer.sample()
        
        # Test size
        self.assertEqual(len(states), self.batch_size)
        self.assertEqual(len(actions), self.batch_size)
        self.assertEqual(len(rewards), self.batch_size)
        self.assertEqual(len(next_states), self.batch_size)
        self.assertEqual(len(dones), self.batch_size)



class AgentTest(TestCase):

    def setUp(self):
        self.state_size = 10
        self.action_size = 4
        self.batch_size = 64
        seed = 40
        self.agent = Agent(state_size=self.state_size, 
                           action_size=self.action_size, 
                           batch_size=self.batch_size,
                           seed=seed, 
                           device=torch.device("cpu"))

    def test_act(self):
        state = np.random.rand(self.state_size).reshape(1, -1)

        act = self.agent.act(state, eps=0.5)
        self.assertIsInstance(act, np.int64)

    def test_evaluate_target(self):
        rewards = torch.rand(self.batch_size).reshape(-1, 1)
        next_states = torch.rand(self.batch_size, self.state_size)
        dones = np.ones(self.batch_size)
        dones[0:int(self.batch_size/3.0 * 2)] = 0
        dones = torch.from_numpy(dones.reshape(-1, 1).astype(bool))
        target = self.agent._evaluate_target(rewards, next_states, dones, 0.9)

        ## Check the terminate case
        assert target.shape == torch.Size([self.batch_size, 1])
        terminate_index = dones.eq(True)
        left, right = target[terminate_index], rewards[terminate_index]
        self.assertTrue(torch.all(torch.eq(left, right)))



    
