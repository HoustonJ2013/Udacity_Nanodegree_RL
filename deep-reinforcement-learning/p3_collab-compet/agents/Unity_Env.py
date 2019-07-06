from unityagents import UnityEnvironment
import numpy as np
import gym


REACHER_STATES_MIN = np.array([ -4,  -4,  -4,  -1,  -1,  -1,  -1, -10,  -2,  -3, -14, -10, -13,
        -9, -10,  -9,  -1,  -1,  -1,  -1, -11, -10,  -6, -19, -20, -18,
        -8,  -1,  -8,  -1,  -1,  -1,  -1])
REACHER_STATES_MAX = np.array([4,  4,  4,  1,  1,  1,  1, 10,  2,  3, 14, 10, 13,  9, 10,  9,  1,
        1,  1,  1, 11, 10,  6, 19, 20, 18,  8,  1,  8,  1,  1,  1,  1])

MOUNTAINCAR_MIN = np.array([-1.2, -0.07])
MOUNTAINCAR_MAX = np.array([0.6, 0.07])

class UnityEnv_Tennis():
    '''
    Simply the syntax for Unity environment, make it similar to gym
    Dedicated for banana environment. The state scaling is hard coded 
    for banana
    '''
    def __init__(self, env_file, states_min=REACHER_STATES_MIN, states_max=REACHER_STATES_MAX):
        
        ## env_file is the unity environment file
        ## states_min min for the state values  (n_states, )
        ## states_max max for the state values  (n_states, )
        self.env = UnityEnvironment(file_name=env_file)
        self.brain_name = self.env.brain_names[0]
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(self.env_info.agents)
        self.brain = self.env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        state = self.env_info.vector_observations[0]
        self.state_size = len(state)
        # self.state_min = states_min
        # self.state_max = states_max

    def reset(self, train_mode=True):
        self.env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        ## return shape self.num_agents x self.state_size
        return self.env_info.vector_observations
        # return self._state_scaler(self.env_info.vector_observations)

    def step(self, action):
        self.env_info = self.env.step(action)[self.brain_name]
        next_state = self.env_info.vector_observations
        reward = self.env_info.rewards
        dones = self.env_info.local_done 
        ## return shape self.num_agents x item size
        return next_state, np.array(reward), np.array(dones), "dummy"

    def render(self):
        return 

    def close(self):
        self.env.close()

    # def _state_scaler(self, state):
    #     ## state self.num_agents x self.state_size
    #     state_scaled = (state - self.state_min) / (self.state_max - self.state_min)
    #     return state_scaled


class Vgym:
    ## A wrapper for gym environment that change the shape of states, rewards, and actions to match the PPO agent
    def __init__(self, env_name):
        self.env = gym.make(env_name) 
        self.state_min = MOUNTAINCAR_MIN
        self.state_max = MOUNTAINCAR_MAX

    def reset(self):
        return self._state_scaler(self.env.reset().reshape(1, -1))

    def render(self):
        self.env.render()
    
    def step(self, action):
        next_state, reward, done, _ = self.env.step(action[0])
        next_state = self._state_scaler(next_state)
        return next_state.reshape(1, -1), np.array([reward]), np.array([done]), np.array([_])

    def _state_scaler(self, state):
        ## state self.num_agents x self.state_size
        state_scaled = (state - self.state_min) / (self.state_max - self.state_min)
        return state_scaled

    


    