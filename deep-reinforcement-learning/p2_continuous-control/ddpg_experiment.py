import numpy as np
import random
import argparse
from collections import namedtuple, deque
from unityagents import UnityEnvironment
import gym
from agents.ddpg_agent import Agent as DDPG_Agent

from agents.Unity_Env import UnityEnv_Reacher, Vgym
from agents.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
from time import gmtime, strftime

def main(args):

    ## Support two environment, LunarLander was used to debug 
    if args.env == "Reacher_unity_v1":
        model_param = {"actor_fc_units": args.actor_fc_units, 
                    "critic_fc_units1": args.critic_fc_units1, 
                    "critic_fc_units2": args.critic_fc_units2, 
                    "critic_fc_units3": args.critic_fc_units3}
        env = UnityEnv_Reacher(env_file="Reacher_Linux/Reacher.x86_64")
        state_size = env.state_size
        action_size = env.action_size

    elif args.env == "Reacher_unity_v2":
        model_param = {"actor_fc_units": args.actor_fc_units, 
                    "critic_fc_units1": args.critic_fc_units1, 
                    "critic_fc_units2": args.critic_fc_units2, 
                    "critic_fc_units3": args.critic_fc_units3}
        env = UnityEnv_Reacher(env_file="Reacher_Linux_20/Reacher.x86_64")
        state_size = env.state_size
        action_size = env.action_size

    elif args.env == "MountainCarContinuous-v0":  ## LunarLanderContinuous-v2
        model_param = {"actor_fc_units": args.actor_fc_units, 
                    "critic_fc_units1": args.critic_fc_units1, 
                    "critic_fc_units2": args.critic_fc_units2, 
                    "critic_fc_units3": args.critic_fc_units3}

        env = Vgym('MountainCarContinuous-v0')
        state_size = env.env.observation_space.shape[0]
        action_size = env.env.action_space.shape[0]

    ## Enviornment description 
    print("Current environment is ", args.env)
    print("State size is %i"%(state_size))
    print("action size is %i"%(action_size))
    print("A typical state looks like", env.reset())


    ## 
    print("Test parameters", args)
    ## device for training
    if args.device.lower() == "cpu".lower():
        device = torch.device("cpu")
        print("Training uses CPU")
    else:
        device = torch.device("cuda:0")
        print("Training uses GPU")

    ## Whether to use prioritized replay buffer
    use_prioritized_replay = args.per
    if args.per:
        print("Using Prioritized Replay")

    
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
                weight_decay=args.weight_decay,
                model_param=model_param, 
                max_t=args.max_t, 
                update_every=args.update_every,
                n_episode_bf_train=args.n_episode_bf_train, ## Train start after n_episode 
                n_episode_stop_explore=args.n_episode_stop_explore, 
                device=device, 
                loss=args.loss,
                )

    ## Train the agent
    n_episodes = args.num_episodes
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=args.score_window_size)  # last 100 scores
    game_length = deque(maxlen=args.score_window_size)

    ## Current time string 
    now_string = strftime("%Y_%m_%d_%H_%M_%S", gmtime()) 

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.n_episode = i_episode
        score = 0
        game_len = 0
        for t in range(args.max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            game_len += 1
            if np.array(done).all():
                break 
        game_length.append(game_len)
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        if agent.n_episode > args.n_episode_bf_train:
            print('\nEpisode {}\t Average Score: {:.2f}  \t Game length {} Critic Running Loss: {:.05f} Actor Running Loss: {:.05f}'.format(i_episode, 
                                                    np.mean(scores_window),  
                                                    game_len, 
                                                    np.mean(agent.critic_running_loss), 
                                                    np.mean(agent.actor_running_loss)), end="")

        if i_episode % args.score_window_size == 0 and agent.n_episode > args.n_episode_bf_train:
            print('\nEpisode {}\tAverage Score: {:.2f} and Average game length {} critic running_loss {} actor running_loss {}'\
                .format(i_episode, np.mean(scores_window), np.mean(game_length), np.mean(agent.critic_running_loss), 
                np.mean(agent.actor_running_loss)))
            check_point_name = args.env + "_episodes_" + str(i_episode) + "_score_" + str(np.mean(scores_window)) + now_string + \
                "_" +  args.testname + "_checkpoint.pth"
            torch.save(agent.actor_local.state_dict(), "models/actor_iter" + str(i_episode) + "_" + check_point_name)
            torch.save(agent.critic_local.state_dict(), "models/critic_iter" + str(i_episode) + "_" + check_point_name)
        if np.mean(scores_window)>=args.score_threshold or i_episode == n_episodes:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            check_point_name = args.env + "_episodes_" + str(i_episode) + "_score_" + str(np.mean(scores_window)) + now_string + \
                "_" +  args.testname + "_checkpoint.pth"
            if args.save_replay:
                print("saving replay buffer ...")
                replay_buffer_name = args.env + now_string + "_replay_buffer_" + args.testname
                replay_buffer = [tem_._asdict() for tem_ in list(agent.memory.memory)]
                with open("models/" + replay_buffer_name, "wb") as pickle_file:
                    pickle.dump(replay_buffer, pickle_file)
                
            
            torch.save(agent.actor_local.state_dict(), "models/actor_" + check_point_name)
            torch.save(agent.critic_local.state_dict(), "models/critic_" + check_point_name)
            
            print("saving models ...")
    
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
    parser.add_argument('--save_replay', action="store_true", default=False)
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--actor_fc_units', type=int, default=128)
    parser.add_argument('--critic_fc_units1', type=int, default=64)
    parser.add_argument('--critic_fc_units2', type=int, default=128)
    parser.add_argument('--critic_fc_units3', type=int, default=64)
    parser.add_argument('--lr_actor', type=float, default=1e-4)
    parser.add_argument('--lr_critic', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--update_every', type=int, default=1)
    parser.add_argument('--score_threshold', type=int, default=30)
    parser.add_argument('--score_window_size', type=int, default=100)

    ## Environment related parameters
    parser.add_argument('--env', type=str, default="Reacher_unity")
    # parser.add_argument('--agent', type=str, default="ddpg")
    parser.add_argument('--n_episode_bf_train', type=int, default=0)
    parser.add_argument('--n_episode_stop_explore', type=int, default=1000)
    parser.add_argument('--num_episodes', type=int, default=2000,
                        help='no. of epoches to train the model')
    parser.add_argument('--max_t', type=int, default=2000)
    parser.add_argument('--loss', type=str, default="mse")

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--testname', type=str, default="")

    args = parser.parse_args()
    main(args)