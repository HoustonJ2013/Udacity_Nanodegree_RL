[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


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

### Introduction: Continuous Control

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

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



