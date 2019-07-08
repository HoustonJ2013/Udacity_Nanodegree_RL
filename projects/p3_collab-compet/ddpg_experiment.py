import numpy as np
import random
import argparse
from collections import namedtuple, deque
from unityagents import UnityEnvironment
import gym
from agents.ddpg_agent import Agent as DDPG_Agent

from agents.Unity_Env import UnityEnv_Tennis
from agents.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
from time import gmtime, strftime


def load_memory(replay_buffer_obj, loaded_pickle_memory):
    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    for item in loaded_pickle_memory:
        item_tf = experience(item["state"], item["action"], item["reward"], item["next_state"], item["done"])
        replay_buffer_obj.memory.append(item_tf)
    return replay_buffer_obj


def main(args):

    ## DDPG to solve stationary mutli-agent tennis problem
    
    model_param = {"actor_fc_units": args.actor_fc_units, 
                "actor_fc_units2": args.actor_fc_units2, 
                "critic_fc_units1": args.critic_fc_units1, 
                "critic_fc_units2": args.critic_fc_units2, 
                "critic_fc_units3": args.critic_fc_units3}
    env = UnityEnv_Tennis(env_file="Tennis_Linux/Tennis.x86_64")
    num_agents = env.num_agents
    state_size = env.state_size
    action_size = env.action_size


    ## Enviornment description 
    print("Num of agents is %i"%(num_agents))
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

    ## Same agent play two roles
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
                n_episode_stop_explore=args.n_episode_stop_explore + args.restart_iter, 
                device=device, 
                loss=args.loss,
                )

    if args.retrain: 
        agent.actor_local.load_state_dict(torch.load(args.actor_weight))
        agent.actor_target.load_state_dict(torch.load(args.actor_weight))
        agent.critic_local.load_state_dict(torch.load(args.critic_weight))
        agent.critic_target.load_state_dict(torch.load(args.critic_weight))
        with open(args.replay_buffer_pickle, "rb") as file:
            replay_buffer_pickle = pickle.load(file)
        load_memory(agent.memory, replay_buffer_pickle)

    ## Train the agent
    n_episodes = args.num_episodes
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=args.score_window_size)  # last 100 scores
    game_length = deque(maxlen=args.score_window_size)

    ## Current time string 
    now_string = strftime("%Y_%m_%d_%H_%M_%S", gmtime()) 
    actor_lr_start = args.lr_actor
    critic_lr_start = args.lr_critic

    for i_episode in range(args.restart_iter, args.restart_iter + n_episodes):
        state = env.reset()
        agent.n_episode = i_episode
        
        score = 0
        game_len = 0
        for t in range(args.max_t):

            ## Agent 
            action1 = agent.act(state[0].reshape(1, -1))
            action2 = agent.act(state[1].reshape(1, -1))
            action = np.concatenate((action1, action2), axis=0)
            next_state, reward, done, _ = env.step(action)
            
            # print(state.shape, state[0].shape, action.shape, reward.shape)

            ## Train from two perspectives 
            ## Train agent 1 perspective
            agent.step(state[0], action[0], reward[0], next_state[0], done[0])
            ## Train agent 2 perspective
            agent.step(state[1], action[1], reward[1], next_state[1], done[1])
            state = next_state
            score += reward
            game_len += 1
            if np.array(done).any():
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
            if args.save_replay:
                print("saving replay buffer ...")
                replay_buffer_name = args.env + now_string + "_replay_buffer_iter" + str(i_episode) + args.testname
                replay_buffer = [tem_._asdict() for tem_ in list(agent.memory.memory)]
                with open("models/" + replay_buffer_name, "wb") as pickle_file:
                    pickle.dump(replay_buffer, pickle_file)

            torch.save(agent.actor_local.state_dict(), "models/actor_iter" + str(i_episode) + "_" + check_point_name)
            torch.save(agent.critic_local.state_dict(), "models/critic_iter" + str(i_episode) + "_" + check_point_name)
        if np.mean(scores_window)>=args.score_threshold or i_episode == args.restart_iter + n_episodes:
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
            break
    
    score_name = args.env + "_episodes_" + str(i_episode) + "_score_" + str(np.mean(scores_window)) + now_string + "_" +  args.testname +  "_scores"
    np.save("models/" + score_name, scores)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Agent related parameters
    parser.add_argument('--per', action="store_true", default=False)
    parser.add_argument('--save_replay', action="store_true", default=False)
    parser.add_argument('--retrain', action="store_true", default=False)
    parser.add_argument('--actor_weight', type=str, default="")
    parser.add_argument('--critic_weight', type=str, default="")
    parser.add_argument('--replay_buffer_pickle', type=str, default="")
    parser.add_argument('--restart_iter', type=int, default=1)
    
    parser.add_argument('--buffer_size', type=int, default=int(1e6))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--actor_fc_units', type=int, default=128)
    parser.add_argument('--actor_fc_units2', type=int, default=64)
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
    parser.add_argument('--env', type=str, default="Tennis")
    # parser.add_argument('--agent', type=str, default="ddpg")
    parser.add_argument('--n_episode_bf_train', type=int, default=200)
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