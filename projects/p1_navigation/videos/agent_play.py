import gym
import sys
import torch
import numpy as np
sys.path.append("../")
from dqn.opencv_avi import record_policy
from time import sleep

## Trained agent
from dqn.dqn_agent import UnityEnv_simple
from dqn.dqn_agent import Agent
env = UnityEnv_simple(env_file="../Banana_Linux/Banana.x86_64")
state_size = env.state_size
action_size = env.action_size
agent_2 = Agent(state_size=state_size, action_size=action_size)
agent_2.qnetwork_local.load_state_dict(torch.load("../models/Banana_unity_episodes_2000_score_15.022019_05_19_20_17_50_banana_per_20_checkpoint.pth"))

for i in range(10):
    sleep(1)
    print(10-i, "secs to start")

game_over = False
max_length = 300
total_reward = 0
step = 0
state = env.reset(train_mode=False)
while (not game_over) and step <= max_length:
    # action = agent_2.act(state) ## Trained agent play
    action = np.random.randint(4)
    state, reward, game_over, _ = env.step(action)
    total_reward += reward
    step += 1

env.close()