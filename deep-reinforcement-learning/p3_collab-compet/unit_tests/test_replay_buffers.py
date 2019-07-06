from unittest import TestCase
import numpy as np
from agents.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
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


class PrioritizedReplayBufferTest(TestCase):

    def setUp(self):
        self.action_size = 1
        self.buffer_size = 1000
        self.batch_size = 64
        seed=42
        self.replaybuffer = PrioritizedReplayBuffer( 
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
                ## Initial weight for each add is 1 
                self.assertEqual(self.replaybuffer.memory.total_p, i_ + 1)
    
    def test_sample(self):
        
        for i_ in range(2 * self.buffer_size):
            state, action, reward, next_state, done = \
            np.array([3, 4, i_]), np.array([2]), 3, np.array([2, 3, 5]), False
            self.replaybuffer.add(state, action, reward, next_state, done)

        tree_idx, batch_memory, ISWeights = self.replaybuffer.sample()
        states, actions, rewards, next_states, dones = batch_memory
        # Test size
        self.assertEqual(actions.shape, torch.Size([self.batch_size, self.action_size]))
        self.assertEqual(rewards.shape, torch.Size([self.batch_size, 1]))
        self.assertEqual(ISWeights.shape, torch.Size([self.batch_size, 1]))
        self.assertEqual(dones.shape, torch.Size([self.batch_size, 1]))
        self.assertEqual(ISWeights.shape, torch.Size([self.batch_size, 1]))

    def test_batch_update(self):
        for i_ in range(2 * self.buffer_size):
            state, action, reward, next_state, done = \
            np.array([3, 4, i_]), np.array([2]), 3, np.array([2, 3, 5]), False
            self.replaybuffer.add(state, action, reward, next_state, done)
        total_p_b = self.replaybuffer.memory.total_p
        tree_idx, batch_memory, ISWeights = self.replaybuffer.sample()
        abs_errors = np.arange(1, self.batch_size + 1, dtype="float") / self.buffer_size
        self.replaybuffer.batch_update(tree_idx, abs_errors)
        tree_idx2, batch_memory, ISWeights = self.replaybuffer.sample()
        total_p_a = self.replaybuffer.memory.total_p
        ## Assert batch update updated the priority
        self.assertTrue(abs(total_p_b - total_p_a) > 0)
        ## Assert the two random sample doesn't give the same tree_idx2
        self.assertTrue((tree_idx != tree_idx2).any()) 