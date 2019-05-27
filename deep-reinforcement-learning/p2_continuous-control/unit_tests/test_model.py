from unittest import TestCase
from agents.model import QNetwork
import torch

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