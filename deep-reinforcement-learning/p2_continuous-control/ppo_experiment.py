import numpy as np
import random
import argparse
from collections import namedtuple, deque
from unityagents import UnityEnvironment
import gym
from agents.ppo_agent import PPOAgent 

from agents.Unity_Env import UnityEnv_Reacher, Vgym
from agents.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
from time import gmtime, strftime

def main(args):

    ## Support two environment, LunarLander was used to debug 
    if args.env == "Reacher_unity":
        model_param = {"actor_fc_units": 64, 
                    "critic_fc_units1": 32, 
                    "critic_fc_units2": 64, 
                    "critic_fc_units3": 64}
        env = UnityEnv_Reacher(env_file="Reacher_Linux/Reacher.x86_64")
        state_size = env.state_size
        action_size = env.action_size

    elif args.env == "Reacher_unity_v2":
        model_param = {"actor_fc_units": 64, 
                    "critic_fc_units1": 32, 
                    "critic_fc_units2": 64, 
                    "critic_fc_units3": 64}
        env = UnityEnv_Reacher(env_file="Reacher_Linux_20/Reacher.x86_64")
        state_size = env.state_size
        action_size = env.action_size

    elif args.env == "MountainCarContinuous-v0":  ## LunarLanderContinuous-v2
        model_param = {"actor_fc_units": 10, 
                    "critic_fc_units1": 2, 
                    "critic_fc_units2": 10, 
                    "critic_fc_units3": 10}

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
    else:
        device = torch.device("cuda:0")

    ## Whether to use prioritized replay buffer
    use_prioritized_replay = args.per
    if args.per:
        print("Using Prioritized Replay")

    
    agent = PPOAgent(
                env=env, 
                state_size=state_size, 
                action_size=action_size, 
                model_param=model_param, 
                random_seed=42,
                batch_size=args.batch_size,
                T_step=1, 
                clip_param=0.2,
                K_epoch=1,
                episodic=True,
                gamma=args.gamma,
                tau=args.tau, 
                lr_actor=args.lr_actor,
                lr_critic=args.lr_critic, 
                weight_decay=args.weight_decay,
                device=device, 
                loss=args.loss,
                )

    ## Train the agent
    n_iterations = args.n_iterations
    scores = []                        # list containing scores from each episode
    
    ## Current time string 
    now_string = strftime("%Y_%m_%d_%H_%M_%S", gmtime()) 

    evaluate_n_iter = args.evaluate_n_iter
    action_std = 1

    for i_iter in range(1, n_iterations+1):

        ## Run roll outs 
        action_std = max(action_std * 0.99, 0.3)
        agent.run_policy(action_std=action_std, n_step=20)
        ## Learng and updates 
        agent.learn(batch_size=256, k_epoch=20)

        print("\rTraining agent iteration %i action std %f training loss actor %f and critic %f"%(i_iter, 
                                                                                    action_std, 
                                                                                    np.mean(agent.actor_running_loss), 
                                                                                    np.mean(agent.critic_running_loss)), flush=True, end="")
 

        if i_iter > 0 and i_iter%evaluate_n_iter == 0 or i_iter == n_iterations:

            score, _, game_lens = agent.evaluation(eval_episodes=100)
            print("\nEvaluation: Average score is %0.3f and average game_len is %i training loss actor %f and critic %f"%(score, int(np.mean(game_lens)), 
                                                                                                            np.mean(agent.actor_running_loss), 
                                                                                                            np.mean(agent.critic_running_loss)))
            scores.append(score)

            if score >=args.score_threshold or i_iter == n_iterations:
                print('Environment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iter, score))
                check_point_name = args.env + "_iter_" + str(i_iter) + "_score_" + str(score) + now_string + \
                    "_" +  args.testname + "_checkpoint.pth"
                torch.save(agent.actor_local.state_dict(), "models/actor_" + check_point_name)
                torch.save(agent.critic_local.state_dict(), "models/critic_" + check_point_name)
                print("saving models ...")
    
    score_name = args.env + "_n_iter_" + str(i_iter) + "_" + now_string + "_" +  args.testname +  "_scores"
    np.save("models/" + score_name, np.array(scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Agent related parameters
    parser.add_argument('--per', action="store_true", default=False)
    parser.add_argument('--save_replay', action="store_true", default=False)
    parser.add_argument('--buffer_size', type=int, default=int(1e4))
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_actor', type=float, default=1e-3)
    parser.add_argument('--lr_critic', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--update_every', type=int, default=1)
    parser.add_argument('--score_threshold', type=int, default=200)
    parser.add_argument('--score_window_size', type=int, default=100)

    ## Environment related parameters
    parser.add_argument('--env', type=str, default="MountainCarContinuous-v0")
    # parser.add_argument('--agent', type=str, default="ddpg")
    parser.add_argument('--n_iterations', type=int, default=200,
                        help='No. of iterations to train')
    parser.add_argument('--evaluate_n_iter', type=int, default=50,
                        help='Evaluate the agent every n iteration')
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=float, default=0.999)
    parser.add_argument('--max_t', type=int, default=2000)
    parser.add_argument('--loss', type=str, default="mse")

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default="cpu")


    parser.add_argument('--testname', type=str, default="")

    args = parser.parse_args()
    print(args)
    main(args)
