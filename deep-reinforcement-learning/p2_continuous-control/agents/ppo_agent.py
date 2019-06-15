## This PPO is designed for continuous control 
## It is implemented for Unity Reacher-V2, which is a version of reacher that simplify the parallelization 
## Current implementation support both episodic and online updates
## For episodic version, run the policy for a whole game and learn
## For the online version, run the policy for a few steps (T default 1) and use Bellman equation to calculate Advantages and update

import numpy as np
import random
import copy
from collections import namedtuple, deque
from . import model, replay_buffers
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as tdist
from copy import deepcopy

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay
UPDATE_EVERY = 4        # how often to update the network
K_epoch = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, env, state_size, action_size, model_param, random_seed=42,
                device=torch.device("cpu"), ## torch.device("cuda:0") or torch.device("cpu")
                batch_size=BATCH_SIZE,
                T_step=1, ## roll out T step for critic accuracy 
                max_t=1000,
                gamma=GAMMA,  
                tau=TAU,
                clip_param=0.2, 
                lr_actor = LR_ACTOR,
                lr_critic = LR_CRITIC,
                K_epoch=3, 
                weight_decay = WEIGHT_DECAY, 
                episodic=True, ## episodic or online
                loss="mse",):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.env = env
        self.env_state = self.env.reset()
        self.state_size = state_size
        self.action_size = action_size
        self.done_penalty=None
        self.batch_size = batch_size
        self.roll_out_n_steps = T_step
        self.T_step = T_step
        self.max_steps = max_t
        self.gamma = gamma
        self.tau = tau
        self.clip_param = clip_param
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.seed = random.seed(random_seed)
        self.K_epoch = K_epoch
        self.device = device
        self.critic_running_loss = deque(maxlen=100)
        self.actor_running_loss = deque(maxlen=100)
        self.critic_loss = loss
        self.episodic = episodic
        self.n_episode = 0
        self.n_steps = 0
        # Actor Network (w/ Target Network) determinstic
        self.actor_local = model.Actor(state_size, action_size, random_seed, fc_units=model_param["actor_fc_units"]).to(self.device)
        self.actor_target = model.Actor(state_size, action_size, random_seed, fc_units=model_param["actor_fc_units"]).to(self.device)
        ## actor distribution
        self.actor_dist = tdist.Normal
        self.action_std = 0.2

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

        # replay buffers
        buffer_size = None
        self.memory = replay_buffers.ReplayBuffer(action_size=action_size, 
                                                 buffer_size=buffer_size, 
                                                 batch_size=batch_size, 
                                                 seed=random_seed, 
                                                 device=device)

    ## Main Step 1: Run Policy 
    def run_policy(self, action_std=0.3, n_step=1):
        self.action_std = action_std
        if self.episodic:
            self.roll_out_n_steps = self.max_steps
        else:
            self.roll_out_n_steps = self.T_step
        
        for i in range(n_step):
            self._interact_n_step(action_std)

    def _deterministic_act(self, state):
        """Returns actions for given state as per current policy."""
        
        state = torch.from_numpy(state).float().to(self.device) ## Use only CPU to roll out play
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state)
        self.actor_local.train()
        return torch.clamp(action, -1, 1)

    def _value_estimate(self, state, action):
        state = torch.from_numpy(state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        self.critic_local.eval()
        with torch.no_grad():
            value = self.critic_local(state, action).cpu().data.numpy()
        self.critic_local.train()
        return value

    def _stochastic_act(self, state, action_std):
        action = self._deterministic_act(state)
        x_dist = self.actor_dist(action, action_std)
        action_s = x_dist.sample().cpu().data.numpy()
        return np.clip(action_s, -1, 1)

    def _interact_n_step(self, action_std):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
        ## state shape batch_size x state_size 
        ## action shape batch_size x action_size
        ## reward shape batch_size x 1
        states = []
        actions = []
        rewards = []
        # take n steps
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = self._stochastic_act(self.env_state, action_std)
            next_state, reward, done, _ = self.env.step(action)
            actions.append(action)
            ## In case any one of the 20 reachers is done
            done_index = done == True
            reward[done_index] = 0
            rewards.append(reward)
            final_state = next_state
            self.env_state = next_state
            if done.all():
                self.env_state = self.env.reset()
                self.n_steps = 0
                break
        # discount reward
        if done.all():
            self.n_episode += 1
            self.episode_done = True
            final_value = np.zeros_like(reward)
        else:
            self.episode_done = False
            final_action = self._stochastic_act(final_state, action_std)
            final_value = self._value_estimate(final_state, final_action)
            done_index = done == True
            final_value[done_index] = 0
        # print(rewards, rewards[0].shape, final_value.shape)
        rewards = self._discount_reward(rewards, final_value)
        self.n_steps += 1

        ## Update memory
        for i in range(len(states)):
            state, action, reward = states[i], actions[i], rewards[i]
            if len(state.shape) > 1:
                for i in range(state.shape[0]):
                    self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])
            else:
                self.memory.add(state, action, reward, next_state, done)

    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value.flatten()
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    ## Main Step 2: Learn
    def learn(self, k_epoch=None, batch_size=None):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        if k_epoch is None:
            k_epoch = self.K_epoch
        if batch_size is None:
            batch_size = self.batch_size
        for i in range(k_epoch):

            ## Sample from memory
            experiences = self.memory.sample(batch_size=batch_size)
            states, actions, rewards, next_states, dones = experiences

            
            ## Update actor 
            self.actor_optimizer.zero_grad()
            advantages = self._advantage(rewards, states, actions)
            action_log_prob = self._action_log_prob(self.actor_local, states, actions)
            action_log_prob_old = self._action_log_prob(self.actor_target, states, actions)
            ratio = torch.exp(action_log_prob - action_log_prob_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages

            # PPO's pessimistic surrogate (L^CLIP)
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            actor_loss.backward()
            self.actor_optimizer.step()

            ##Update Critic, here the rewards is reward-to-go. 
            self.critic_optimizer.zero_grad()
            target_values = rewards
            values = self.critic_local(states, actions)
            if self.critic_loss == "huber":
                critic_loss = torch.nn.SmoothL1Loss(values, target_values)
            else:
                critic_loss = torch.nn.MSELoss()(values, target_values)
            critic_loss.backward()
            self.critic_optimizer.step()

            self.critic_running_loss.append(critic_loss.item())
            self.actor_running_loss.append(actor_loss.item())

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau)     
        self.memory.reset()

    def _advantage(self, rewards, states, actions):
        values = self.critic_target.forward(states, actions)
        advantages = rewards - values
        return advantages                 

    def _action_log_prob(self, action_net_work, states, actions):
        pred_actions = action_net_work.forward(states)
        x_dist = self.actor_dist(pred_actions, self.action_std)
        log_prob = x_dist.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return log_prob

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

    def evaluation(self, env=None, eval_episodes=100, action_std=0.01):
        '''
        Evaluate the agent for given number of episodes
        '''
        scores = []
        rewards = []
        game_lens = []
        state = self.env.reset()
        n_agents = state.shape[0]
        n_iterations = int(eval_episodes / float(n_agents)) + 1
        print("Evaluate %i iterations"%(n_iterations))
        if env is None:
            env = self.env

        for i in range(n_iterations):
            rewards_i = []
            state = env.reset()
            action = self._stochastic_act(state, action_std=action_std)
            state, reward, done, info = env.step(action)
            rewards_i.append(reward)
            game_len = 0
            while not done.all():
                action = self._stochastic_act(state, action_std=action_std)
                next_state, reward, done, info = env.step(action)
                state = next_state
                rewards_i.append(reward)
                # print("iter", i, "reward", reward)
                game_len += 1
            rewards.append(rewards_i)
            game_lens.append(game_len)
            rewards_i = np.array(rewards_i)
            # print("reward_i shape", rewards_i.shape)
            rewards_i_sum = np.sum(rewards_i, axis=0)
            # print("reward_i_sum shape", rewards_i_sum.shape)
            scores.append(np.mean(rewards_i_sum))
        # print("Score is", scores)
        self.env_state = env.reset()
        return np.mean(scores), rewards, game_lens

# class PPOReplayBuffer:

#     def __init__(self, device):
#         """Initialize a ReplayBuffer object.

#         Params
#         ======
#         """
#         self.memory = []  
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#         self.device = device
#         random.seed(seed)
    
#     def add(self, state, action, reward, next_state, done):
#         """Add a new experience to memory."""
#         e = self.experience(state, action, reward, next_state, done)
#         self.memory.append(e)
    
#     def all(self):
#         """Randomly sample a batch of experiences from memory."""
#         experiences = self.memory

#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
#         return (states, actions, rewards, next_states, dones)

#     def reset(self):
#         self.memory = []

#     def __len__(self):
#         """Return the current size of internal memory."""
#         return len(self.memory)

