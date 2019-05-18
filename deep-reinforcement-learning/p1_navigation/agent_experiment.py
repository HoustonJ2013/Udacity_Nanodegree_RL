import numpy as np
import random
import argparse
from collections import namedtuple, deque
from unityagents import UnityEnvironment
import gym
from dqn.dqn_agent import UnityEnvironment, Agent, UnityEnv_simple, ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle 

def main(args):

    ## Support two environment, LunarLander was used to debug 
    if args.env == "LunarLander-v2":
        env = gym.make(args.env)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
    elif args.env == "Banana_unity":
        env = UnityEnv_simple(env_file="Banana_Linux/Banana.x86_64")
        state_size = env.state_size
        action_size = env.action_size
    
    ## Enviornment description 
    print("Current environment is ", args.env)
    print("State size is %i"%(state_size))
    print("action size is %i"%(action_size))
    print("A typical state looks like", env.reset())

    ## device for training
    if args.device.lower() == "cpu".lower():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    agent = Agent(state_size=state_size, 
                  action_size=action_size, 
                  seed=42,
                  buffer_size=args.buffer_size,
                  batch_size=args.batch_size,
                  gamma=args.gamma,
                  tau=args.tau, 
                  lr=args.lr,
                  update_every=args.update_every,
                  device=device, 
                  loss=args.loss,
                 )
    # watch an untrained agent
    # state = env.reset()
    # for j in range(200):
    #     action = agent.act(state)
    #     env.render()
    #     state, reward, done, _ = env.step(action)
    #     if done:
    #         break 
    # env.close()

    ## Train the agent
    n_episodes = args.num_episodes
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=args.score_window_size)  # last 100 scores
    game_length = deque(maxlen=args.score_window_size)
    eps = args.eps_start                    # initialize epsilon
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
        if np.mean(scores_window)>=args.score_threshold:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} current eps :{:.4f}'.format(i_episode-100, np.mean(scores_window), eps))
            check_point_name = args.env + "_episodes_", str(i_episode) + "_score_" + str(np.mean(scores_window)) + "_checkpoint.pth"
            torch.save(agent.qnetwork_local.state_dict(), "models/" + check_point_name)
            break
    score_name = args.env + "_episodes_" + str(i_episode) + "_score_" + str(np.mean(scores_window)) + "_scores"
    np.save("models/" + score_name, scores)
    replay_buffer_name = args.env + "_replay_buffer"
    replay_buffer = [tem_._asdict() for tem_ in list(agent.memory.memory)]
    with open("models/" + replay_buffer_name, "wb") as pickle_file:
        pickle.dump(replay_buffer, pickle_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Agent related parameters
    parser.add_argument('--buffer_size', type=int, default=int(1e5))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--update_every', type=int, default=4)
    parser.add_argument('--score_threshold', type=int, default=200)
    parser.add_argument('--score_window_size', type=int, default=100)

    ## Environment related parameters
    parser.add_argument('--env', type=str, default="LunarLander-v2")
    parser.add_argument('--num_episodes', type=int, default=2000,
                        help='no. of epoches to train the model')
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=float, default=0.999)
    parser.add_argument('--max_t', type=int, default=1000)
    parser.add_argument('--loss', type=str, default="mse")

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default="cpu")

    # parser.add_argument('--log_dir', type=str,
    #                     default="../logs/",
    #                     help='Path the input image')

    # parser.add_argument('-m', '--model', type=str, default='segmentation_model_t1',
    #                     help='choose which model to use',
    #                     choices=['resnet50_modified2',
    #                              'segmentation_model_t1',
    #                              ])
    # parser.add_argument('--model_summary', default=False)
    # parser.add_argument('--model_name', type=str, default="explore_australia_t1")
    # parser.add_argument('--loss', type=str, default="dice_loss")
    # parser.add_argument('--rotate_range', type=str, default=None)
    # parser.add_argument('--model_weights', type=str,
    #                     default=None,
    #                     help='Path the input image')
    args = parser.parse_args()
    print(args)
    main(args)
