import numpy as np
import random
import argparse
from collections import namedtuple, deque
from unityagents import UnityEnvironment
import gym
from . import model 
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, 
                device, ## torch.device("cuda:0") or torch.device("cpu")
                buffer_size=BUFFER_SIZE, 
                batch_size=BATCH_SIZE,
                gamma=GAMMA,  
                tau=TAU,
                lr = LR,
                update_every=UPDATE_EVERY, 
                loss="mse",
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.random = random.seed(seed)

        # Q-Network
        self.device = device
        self.qnetwork_local = model.QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = model.QNetwork(state_size, action_size, seed).to(device)
        if loss == "mse":
            self.criteron = torch.nn.MSELoss()
        elif "huber" in loss:
            self.criteron = torch.nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.running_loss = deque(maxlen=100)
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## Check shape of tensor
        # assert states.shape == torch.Size([self.batch_size, self.state_size])
        # assert actions.shape == torch.Size([self.batch_size, 1])
        # assert rewards.shape == torch.Size([self.batch_size, 1])

        target = self._evaluate_target(rewards, next_states, dones, gamma) ## Target Shape batch_size, 1
        self.optimizer.zero_grad()

        output = self.qnetwork_local.forward(states) ## Output is batch_size x n_actions
        Q_net = output[torch.arange(self.batch_size), actions.flatten()].reshape(-1, 1)
        # assert Q_net.shape == torch.Size([self.batch_size, 1])

        loss = self.criteron(Q_net, target)
        loss_before = loss.item()
        loss.backward()
        self.optimizer.step()

        self.running_loss.append(loss.item())

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)           

    def _evaluate_target(self, rewards, next_states, dones, gamma):
        ## Evaluate target
        self.qnetwork_target.eval()
        with torch.no_grad():
            action_next_states = self.qnetwork_target(next_states)
        target = rewards + gamma * action_next_states.max(dim=1).values.reshape(-1, 1)
        terminate_index = dones.eq(True)  ## Index when state is terminated
        target[terminate_index] = rewards[terminate_index]
        return target


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class UnityEnv_simple():
    '''
    Simply the syntax for Unity environment, make it similar to gym
    '''
    def __init__(self, env_file):
        self.env = UnityEnvironment(file_name=env_file)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations[0]
        self.state_size = len(state)
        self.state_min = np.zeros(self.state_size)
        self.state_min[-2:] = [-4, -12.5]
        self.state_max = np.ones(self.state_size)
        self.state_max[-2:] = [4, 12.5]

    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        return self._state_scaler(env_info.vector_observations[0])

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]  
        done = env_info.local_done[0] 
        return self._state_scaler(next_state), reward, done, "dummy"

    def render(self):
        return 

    def close(self):
        self.env.close()

    def _state_scaler(self, state):
        state_scaled = (state - self.state_min) / (self.state_max - self.state_min)
        return state_scaled
