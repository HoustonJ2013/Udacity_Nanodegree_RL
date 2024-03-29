import numpy as np
import random
import copy
from collections import namedtuple, deque
from . import model, replay_buffers

import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay
N_EPISODE_BF_TRAIN = 100
UPDATE_EVERY = 4        # how often to update the network
N_EPISODE_STOP_EXPLORE = 500
MAX_T = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, model_param, random_seed=42, 
                device=torch.device("cpu"), ## torch.device("cuda:0") or torch.device("cpu")
                buffer_size=BUFFER_SIZE, 
                batch_size=BATCH_SIZE,
                gamma=GAMMA,  
                tau=TAU,
                lr_actor = LR_ACTOR,
                lr_critic = LR_CRITIC,
                update_every=UPDATE_EVERY, 
                weight_decay = WEIGHT_DECAY, 
                n_episode_bf_train=N_EPISODE_BF_TRAIN, 
                n_episode_stop_explore=N_EPISODE_STOP_EXPLORE,
                max_t=MAX_T, 
                per=False, # Prioritized experience replay
                loss="mse",):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.seed = random.seed(random_seed)
        self.per = per
        self.device = device
        self.critic_running_loss = deque(maxlen=100 * max_t)
        self.actor_running_loss = deque(maxlen=100 * max_t)
        self.n_episode = 0
        self.n_episode_bf_train = n_episode_bf_train
        self.n_episode_stop_explore = n_episode_stop_explore
        self.update_every = update_every

        # Actor Network (w/ Target Network)
        self.actor_local = model.Actor(state_size, action_size, random_seed, fc_units=model_param["actor_fc_units"], 
                                                                            fc_units2=model_param["actor_fc_units2"]).to(self.device)
        self.actor_target = model.Actor(state_size, action_size, random_seed, fc_units=model_param["actor_fc_units"], 
                                                                            fc_units2=model_param["actor_fc_units2"]).to(self.device)

        ## Check if local and target start with same weights
        for target_param, local_param in zip(self.actor_local.parameters(), self.actor_target.parameters()):
            assert (target_param.data == local_param.data).all()
        print("Actor weights initialized the same between local and target")
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = model.Critic(state_size, action_size, random_seed, fcs1_units=model_param["critic_fc_units1"], 
                                            fc2_units=model_param["critic_fc_units2"], fc3_units=model_param["critic_fc_units3"]).to(self.device)
        self.critic_target = model.Critic(state_size, action_size, random_seed, fcs1_units=model_param["critic_fc_units1"], 
                                            fc2_units=model_param["critic_fc_units2"], fc3_units=model_param["critic_fc_units3"]).to(self.device)
        ## Check if local and target start with same weights
        for target_param, local_param in zip(self.critic_local.parameters(), self.critic_target.parameters()):
            assert (target_param.data == local_param.data).all()
        print("Critic weights initialized the same between local and target")

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        if self.per: 
            self.memory = replay_buffers.PrioritizedReplayBuffer(action_size, buffer_size, batch_size, random_seed, device)
            print("PrioritzedReplayBuffer is instantiated")
        else: 
            self.memory = replay_buffers.ReplayBuffer(action_size, buffer_size, batch_size, random_seed, device)
            print("ReplayBuffer is instantiated")
    
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        if len(state.shape) > 1:
            for i in range(state.shape[0]):
                self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])
        else:
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        self.t_step = (self.t_step + 1) % self.update_every
        if self.n_episode > self.n_episode_bf_train and self.t_step  == 0 and len(self.memory) > self.batch_size:
            self.learn(self.gamma)

    def act(self, state, add_noise=True, clip=1):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise and self.n_episode < self.n_episode_stop_explore:
            action += self.noise.sample()
            return np.clip(action, -clip, clip)
        else:
            return np.clip(action, -clip, clip)

    def reset(self):
        self.noise.reset()

    def learn(self, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            gamma (float): discount factor
        """
        if self.per:
            tree_idx, batch_memory, ISWeights = self.memory.sample()
            states, actions, rewards, next_states, dones = batch_memory
        else: 
            states, actions, rewards, next_states, dones = self.memory.sample()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        if self.per: 
            abs_error = self._abs_error(Q_expected, Q_targets).to(torch.device("cpu")).data.numpy()
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            self.memory.batch_update(tree_idx, abs_error)
            assert np.sum(abs_error) != 0
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()  ## Maximize the Q values
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_running_loss.append(critic_loss.item())
        self.actor_running_loss.append(actor_loss.item())
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def _abs_error(self, qnet, target):
        # qnet target have shape shape batch size x 1,
        return torch.mean(torch.abs(qnet - target), 1) 
    
    def _weighted_mse_loss(self, qnet, target, weights):
        return torch.mean(weights * (qnet - target)**2 )

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

