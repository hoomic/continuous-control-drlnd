# Project 2: Continuous Control

### Introduction

Note: Much of the code/documentation in this repository comes directly from exercises and code provided to me through Udacity's deep reinforcement learning nanodegree program 

This repository provides an agent that learns to control a robotic arm to touch a floating ball!

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

### Instructions

#### Installation
1. To set up the dependencies, follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)

2. clone this repository

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
  (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

  (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

4. Place the file in this repository and unzip (or decompress) the file. 

#### Training the Agent

##### Command Line
To train the agent, simply run:
```python ./continuous_control.py```

If you want to watch the agent as it learns, run:
```python ./continuous_control.py --visualize```

If you want to load the pretrained model included in this repository, run:
```python ./continuous_control.py --load```

If you want to watch one episode of the agent without training, run:
```python ./continuous_control.py --no_train --watch_one_episode```

##### Jupyter Notebook
There is also a Jupyter notebook saved as Continuous_Control.ipynb if you would prefer to train the agent there.

#### The Model
The model used to solve this environment is [DDPG](https://arxiv.org/abs/1509.02971) which is generally considered an "Actor-Critic" method, but some classify it as a DQN for continuous action spaces.

##### Hyperparameters
There are several hyperparameters involved in training this agent. Some of these are controlled from the command line, and others are defined in ```ddpg_agent.py```.

To see all the options available from the command line, run:
```python ./continuous_control.py --help```
