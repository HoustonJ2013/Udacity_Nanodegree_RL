import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
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
        self.linear2 = nn.Linear(20, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x))
        return x
