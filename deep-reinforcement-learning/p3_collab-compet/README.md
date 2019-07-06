[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


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

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.



# 3. Project Folder Stuctures
- Tennis_Linux/   Evironment for linux
- agents/ source codes for DDPG and PPO. The submitted solution was solved by DDPG
- models/ scores log and solved nn models
- notebook/ analysis notebooks for environment and results
- logs/ training logs for model and hyper parameter tests 
- pics/ gif for display

# 4. Run Test

## Run Unit Test
```
python -m unittest discover unit_tests -v
```

## To experiment with different parameters and environment
```
python ddpg_experiment.py --device gpu 

```

