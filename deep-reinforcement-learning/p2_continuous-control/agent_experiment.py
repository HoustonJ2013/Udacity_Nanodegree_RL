import numpy as np
import random
import argparse
from collections import namedtuple, deque
from unityagents import UnityEnvironment
import gym
from agents.ddpg_agent import Agent as DDPG_Agent
from agents.dqn_agent import Agent as DQN_Agent

from agents.Unity_Env import UnityEnv_Reacher
from agents.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
from time import gmtime, strftime

def main(args):

    ## Support two environment, LunarLander was used to debug 
    if args.env == "Reacher_unity":
        env = UnityEnv_Reacher(env_file="Reacher_Linux/Reacher.x86_64")
        state_size = env.state_size
        action_size = env.action_size

    elif args.env == "Reacher_unity_v2":
        env = UnityEnv_Reacher(env_file="Reacher_Linux_20/Reacher.x86_64")
        state_size = env.state_size
        action_size = env.action_size

    elif args.env == "BipedalWalker-v2":
        env = gym.make('BipedalWalker-v2') 
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

    ## Enviornment description 
    print("Current environment is ", args.env)
    print("Selecte agent to solve the env is", args.agent)
    print("State size is %i"%(state_size))
    print("action size is %i"%(action_size))
    print("A typical state looks like", env.reset())

    ## device for training
    if args.device.lower() == "cpu".lower():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    ## Whether to use prioritized replay buffer
    use_prioritized_replay = args.per
    if args.per:
        print("Using Prioritized Replay")

    if args.agent == "ddpg":
        agent = DDPG_Agent(state_size=state_size, 
                    action_size=action_size, 
                    random_seed=42,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    tau=args.tau, 
                    lr_actor=args.lr_actor,
                    lr_critic=args.lr_critic, 
                    per=use_prioritized_replay, 
                    device=device, 
                    loss=args.loss,
                    )
    elif args.agent == "dqn":
        agent = DQN_Agent(state_size=state_size, 
                  action_size=action_size, 
                  seed=42,
                  buffer_size=args.buffer_size,
                  batch_size=args.batch_size,
                  gamma=args.gamma,
                  tau=args.tau, 
                  lr=args.lr,
                  update_every=args.update_every,
                  per=use_prioritized_replay, 
                  device=device, 
                  loss=args.loss,
                 )

    ## Train the agent
    n_episodes = args.num_episodes
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=args.score_window_size)  # last 100 scores
    game_length = deque(maxlen=args.score_window_size)
    eps = args.eps_start                    # initialize epsilon

    ## Current time string 
    now_string = strftime("%Y_%m_%d_%H_%M_%S", gmtime()) 

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        game_len = 0
        for t in range(args.max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            game_len += 1
            if done:
                break 
        game_length.append(game_len)
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(args.eps_end, args.eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f} current eps :{:.4f} Q Network Running Loss: {:.05f}'.format(i_episode, 
                                                np.mean(scores_window), 
                                                eps, 
                                                np.mean(agent.running_loss)), end="")
        if i_episode % args.score_window_size == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} and Average game length {} current eps {:.04f} running_loss {}'\
                .format(i_episode, np.mean(scores_window), np.mean(game_length), eps, np.mean(agent.running_loss)))
        if np.mean(scores_window)>=args.score_threshold or i_episode == n_episodes:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} current eps :{:.4f}'.format(i_episode-100, np.mean(scores_window), eps))
            check_point_name = args.env + "_episodes_" + str(i_episode) + "_score_" + str(np.mean(scores_window)) + now_string + \
                "_" +  args.testname + "_checkpoint.pth"
            if args.agent == "ddpg":
                torch.save(agent.actor_local.state_dict(), "models/actor_" + check_point_name)
                torch.save(agent.critic_local.state_dict(), "models/critic_" + check_point_name)
            elif args.agent == "dqn":
                torch.save(agent.qnetwork_local.state_dict(), "models/" + check_point_name)
            break
    
    score_name = args.env + "_episodes_" + str(i_episode) + "_score_" + str(np.mean(scores_window)) + now_string + "_" +  args.testname +  "_scores"
    np.save("models/" + score_name, scores)
    # replay_buffer_name = args.env + now_string + "_replay_buffer_" + args.testname
    # replay_buffer = [tem_._asdict() for tem_ in list(agent.memory.memory)]
    # with open("models/" + replay_buffer_name, "wb") as pickle_file:
    #     pickle.dump(replay_buffer, pickle_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Agent related parameters
    parser.add_argument('--per', action="store_true", default=False)
    parser.add_argument('--buffer_size', type=int, default=int(1e4))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_actor', type=float, default=1e-4)
    parser.add_argument('--lr_critic', type=float, default=3e-4)
    parser.add_argument('--update_every', type=int, default=4)
    parser.add_argument('--score_threshold', type=int, default=200)
    parser.add_argument('--score_window_size', type=int, default=100)

    ## Environment related parameters
    parser.add_argument('--env', type=str, default="BipedalWalker-v2")
    parser.add_argument('--agent', type=str, default="ddpg")
    parser.add_argument('--num_episodes', type=int, default=2000,
                        help='no. of epoches to train the model')
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=float, default=0.999)
    parser.add_argument('--max_t', type=int, default=1000)
    parser.add_argument('--loss', type=str, default="mse")

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default="cpu")

    parser.add_argument('--testname', type=str, default="")

    args = parser.parse_args()
    print(args)
    main(args)
