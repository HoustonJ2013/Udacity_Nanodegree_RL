import numpy as np
import random
import argparse
from collections import namedtuple, deque
from unityagents import UnityEnvironment
import gym
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, 
                buffer_size=BUFFER_SIZE, 
                batch_size=BATCH_SIZE,
                gamma=GAMMA,  
                tau=TAU,
                lr = LR,
                update_every=UPDATE_EVERY  
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
        self.seed = random.seed(seed)

        # Q-Network
        print(device)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.criteron = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.running_loss = 0
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
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

        ## TODO: compute and minimize the loss
        self.qnetwork_target.eval()
        with torch.no_grad():
            action_next_states = self.qnetwork_target(next_states)
        
        self.optimizer.zero_grad()
        target = rewards + gamma * action_next_states.max(dim=1)
        terminate_index = (dones == True)  ## Index when state is terminated
        target[terminate_index] = rewards[terminate_index]
        output = self.qnetwork_local.forward(states) ## Output is batch_size x n_actions
        Q_net = output[torch.arange(BATCH_SIZE), actions]
        loss = self.criteron(Q_net, target)
        loss.backward()
        self.optimizer.step()
        self.running_loss += loss.item()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

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

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
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

    def reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations[0]

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]  
        done = env_info.local_done[0] 
        return next_state, reward, done, "dummy"

def main(args):
    if args.env == "LunarLander-v2":
        env = gym.make(args.env)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
    elif args.env == "Banan_unity":
        env = UnityEnv_simple(env_file="Banana_Linux/Banana.x86_64")
        state_size = env.state_size
        action_size = env.action_size
    agent = Agent(state_size=state_size, 
                  action_size=action_size, 
                  seed=42,
                  buffer_size=args.buffer_size,
                  batch_size=args.batch_size,
                  gamma=args.gamma,
                  tau=args.tau, 
                  lr=args.lr,
                  update_every=args.update_every
                 )
    # watch an untrained agent
    state = env.reset()
    for j in range(200):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break 
    env.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="LunarLander-v2")
    parser.add_argument('--buffer_size', type=int, default=int(1e5))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--update_every', type=int, default=4)


    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--num_epoch', type=int, default=10,
                        help='no. of epoches to train the model')
    parser.add_argument('--log_dir', type=str,
                        default="../logs/",
                        help='Path the input image')

    parser.add_argument('-m', '--model', type=str, default='segmentation_model_t1',
                        help='choose which model to use',
                        choices=['resnet50_modified2',
                                 'segmentation_model_t1',
                                 ])
    parser.add_argument('--model_summary', default=False)
    parser.add_argument('--model_name', type=str, default="explore_australia_t1")
    parser.add_argument('--loss', type=str, default="dice_loss")
    parser.add_argument('--rotate_range', type=str, default=None)
    parser.add_argument('--model_weights', type=str,
                        default=None,
                        help='Path the input image')
    args = parser.parse_args()
    print(args)
    main(args)
