from unittest import TestCase
import numpy as np
from agents.ppo_agent import PPOAgent
from agents.Unity_Env import Vgym
import torch
import gym

model_param = {"actor_fc_units": 10, 
                    "critic_fc_units1": 2, 
                    "critic_fc_units2": 10, 
                    "critic_fc_units3": 10}

class AgentTest(TestCase):

    def setUp(self):
        self.batch_size = 64
        env = Vgym('MountainCarContinuous-v0')
        self.state_size = env.env.observation_space.shape[0]
        self.action_size = env.env.action_space.shape[0]
        seed = 40
        self.agent = PPOAgent(env=env, state_size=self.state_size, 
                           action_size=self.action_size, 
                           batch_size=self.batch_size,
                           model_param=model_param, 
                           random_seed=seed, 
                           device=torch.device("cpu"))

    def test_act(self):
        state = np.random.random((100, self.state_size))
        act1 = self.agent._deterministic_act(state)
        self.assertTrue(act1.dtype, np.float32)
        self.assertEqual(act1.shape, (100, self.action_size))
        act2 = self.agent._stochastic_act(state, action_std=0.2)
        self.assertTrue(act2.dtype, np.float32)
        self.assertEqual(act2.shape, (100, self.action_size))
        act2_b = self.agent._stochastic_act(state, 0.2)
        self.assertTrue(not (act2==act2_b).all())

        act2 = self.agent._stochastic_act(state, action_std=0.01)
        self.assertTrue(act2.dtype, np.float32)
        self.assertEqual(act2.shape, (100, self.action_size))
        act2_b = self.agent._stochastic_act(state, 0.01)
        self.assertTrue(not (act2==act2_b).all())

    def test_discount_reward(self):
        self.agent.gamma = 0.9    
        rewards = [np.ones((10, 1))] * 4
        rewards = np.array(rewards)
        rewards[:, 9, :] = 2
        final_value = np.zeros((10, 1))
        discount_reward = self.agent._discount_reward(rewards, final_value)
        true_discount_reward = np.zeros_like(rewards)
        true_discount_reward[0] = 3.439 
        true_discount_reward[1] = 2.71
        true_discount_reward[2] = 1.9
        true_discount_reward[3] = 1
        true_discount_reward[0, 9, :] = 6.878
        true_discount_reward[1, 9, :] = 5.42
        true_discount_reward[2, 9, :] = 3.8
        true_discount_reward[3, 9, :] = 2
        # print("calculated discount reward", discount_reward, discount_reward.shape)
        # print("True discount reward", true_discount_reward, true_discount_reward.shape)
        self.assertTrue((true_discount_reward == discount_reward).all())

    def test_interact_n_step(self):
        self.agent._interact_n_step(action_std=0.2)
        self.assertTrue(len(self.agent.memory) > 0)

    def test_run_policy(self):
        self.agent.run_policy(action_std=0.2)
        self.assertEqual(self.agent.action_std, 0.2)

    def test_learn(self):
        self.agent.run_policy(action_std=0.2)
        self.agent.learn()

    def test_evaluation(self):
        self.agent.evaluation(eval_episodes=2)
        pass

    





