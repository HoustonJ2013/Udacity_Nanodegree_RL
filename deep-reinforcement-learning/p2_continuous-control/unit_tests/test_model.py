from unittest import TestCase
from agents.model import QNetwork, Actor, Critic
import torch
import numpy as np

class QNetworkTest(TestCase):

    def setUp(self):
        self.state_size = 10
        self.action_size = 4
        seed = 40
        self.qnetwork = QNetwork(state_size=self.state_size, 
                                 action_size=self.action_size, 
                                 seed=seed)

    def test_init(self):
        self.assertEqual(self.state_size, self.qnetwork.state_size)
        self.assertEqual(self.action_size, self.qnetwork.action_size)

    def test_forward(self):
        state = torch.rand(self.state_size).reshape(1, -1)
        act = self.qnetwork.forward(state)
        self.assertEqual(act.shape, torch.Size([1, self.action_size]))

    def test_functioncall(self):
        state = torch.rand(10).reshape(1, -1)
        act = self.qnetwork(state)
        self.assertEqual(act.shape, torch.Size([1, self.action_size]))

   
class ActorTest(TestCase):

    def setUp(self):
        self.state_size = 10
        self.action_size = 4
        seed = 40
        self.actor = Actor(state_size=self.state_size, 
                                 action_size=self.action_size, 
                                 seed=seed)

    def test_forward(self):
        state = torch.rand((100, self.state_size))
        action = self.actor.forward(state)
        self.assertEqual(action.dtype, torch.float32)
        self.assertEqual(action.shape, (100, self.action_size))
        self.assertTrue(np.min(action.data.numpy()) >= -1)
        self.assertTrue(np.max(action.data.numpy()) <= 1)


class CriticTest(TestCase):

    def setUp(self):
        self.state_size = 10
        self.action_size = 4
        seed = 40
        self.critic = Critic(state_size=self.state_size, 
                                 action_size=self.action_size, 
                                 seed=seed)

    def test_forward(self):
        states = torch.rand((100, self.state_size))
        actions = torch.rand((100, self.action_size))
        output = self.critic.forward(states, actions).data.numpy()
        self.assertEqual(output.shape, (100, 1))