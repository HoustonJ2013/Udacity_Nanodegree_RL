import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, model="linear_model_2"):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(state_size, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, self.action_size)
        self.model_name = model

    def forward(self, state):
        """Build a network that maps state -> action values."""
        if self.model_name == "linear_model_2":
            return self._linearmodel2(state)

    def _linearmodel2(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

