[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# 1. Environment Installation 

## This project was carried out on local workstation with Ubuntu 16.04
Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz

2 GTX 1070

## Python Environment: Create anaconda environment

conda update conda

conda create -n rlearning python=3.6 anaconda

source activate rlearning

## Install Unity and ml-agents

The package provided in the ../python keep giving errors, I followed this [blog](https://alexisrozhkov.github.io/unity_rl/) for installing Unity and ml-agents. 

Installed ml-agents from the github [source code](https://github.com/Unity-Technologies/ml-agents), reverted back to 0.4.0. 


## Pytorch-gpu installation through anaconda

conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

## Setup Note
In order to run LunarLander-v2
```
pip install gym==0.10.8
conda install swig 
pip install box2d-py
```

# 2. Environment Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

# 3. Project Folder Stuctures
- Banana_Linux/   Banana environment for linux
- dqn/ source code for DQN agent, replay buffers and nn models
- models/ scores log and solved nn models
- notebook/ analysis notebooks for environment and results 
- vidoes/ gif for display

# 4. Run Test

## Run Unit Test
```
python -m unittest discover unit_tests -v
```

## To experiment with different parameters and environment
```
python agent_experiment.py --device gpu --env Banana_unity ## Ordinary replaybuffer

python agent_experiment.py --device gpu --per --env Banana_unity ## Prioritized Replay Buffer

python agent_experiment.py --device gpu --env Banana_unity --max_t 1000 --batch_size 64 --num_episodes 200 --score_threshold 13 --score_window_size 5 --update_every 10 --eps_decay 0.99 ## Final parameters for the submission
```



