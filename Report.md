# Learning Algorithm

This repository contains 3 main files that allow the agent to learn to reach out and touch an object

### model.py
This file contains the class definitions for the Actor and Critic networks used in DDPG. The classes derive from torch.nn.Module and they define the neural network in pytorch. 

The class Actor is a fully connected neural network that is defined by its hidden layer sizes. Actor can take a list of arbitrary length that contains the sizes for each hidden layer. Each layer in this network is followed by a ReLU activation function except for the output layer. The Actor takes the state as input and outputs a vector of actions that have a Tanh activation applied to them to so that their values lie in (-1, 1).

The class Critic is a fully connected neural network that is defined by its hidden layer sizes and the layer that the action vector is concatenated to. Similar to the Actor, the Critic can take a list of arbitrary length that contains the sizes of each hidden layer. The Critic takes the state as an input, then concatenates the action vector to one of the hidden layers determined by the parameter action_cat_layer. The Critic outputs a scalar representing the expected value of taking its action in that state.

### ddpg_agent.py
This file contains the hyperparameters for the agent as well as the class definitions for the agent and the experience replay buffer.

The class Agent contains two Actor-Critic Networks One of these networks is the local, and the other is the target. The local network is updated on every step once there are enough samples in the replay buffer to make a batch. The target network is updated every UPDATE_EVERY steps so that it is sightly closer to the local version by the following equation: 

```target_parameters = TAU * local_parameters + (1 - TAU) * target_parameters```

The step() function is where an experience is added to the replay buffer.

The act() function takes a state and outputs the action determined by the local Actor

The learn() function is where the losses are defined and backpropagated for the Actor and the Critic. This is also where the soft update of the target parameters to the local parameters occurs.

The class OUNoise creates random noise from an Ornstein-Uhlenbeck process

The class ReplayBuffer keeps a memory of the last BUFFER_SIZE experiences and allows the agent to randomly sample from these experiences.

### continuous_control.py
This file contains the training loop for the agent. This is where the interaction between the agent and the environment actually takes place. 

### Hyperparameters
The following hyperparameters were used to achieve an average score over 30 in 243 episodes:
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.9             # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-5        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

Some hyperparameters are passed directly to the Agent object:
actor_hidden_layers = [1024, 1024]     # hidden layer sizes for the Actor
critic_hidden_layers = [1024, 1024]    # hidden layer sizes for the Critic
ou_theta = 0.2                       # Ornstein-Uhlenbeck initial theta parameter
ou_sigma = 0.05                       # Ornstein-Uhlenbeck initial sigma parameter
ou_theta_decay = 1.0                  # multiplicative decay applied to ou_theta after every episode
ou_sigma_decay = 1.0                  # multiplicative decay applied to ou_sigma after every episode
energy_penalty = 0.0                  # weight for L1 loss on action vector to reinforce smoother movements

I found that the GAMMA hyperparameter was critically important to getting the agent to solve the task. For many of my attempts, I was using a GAMMA = 0.99 or 0.999. What I found was that the agent would see the object in one part of the space and then wait for it at a point where the object was going to be. Then it would never learn to actually go to that space and touch it the whole time. After observing this behavior, I thought I should lower GAMMA so that the agent was not rewarded as much for doing this.

Using the decay parameters for the Ornstein-Uhlenbeck process did not yield better training times, so they were set to 1.0.

The energy penalty did not help for initial training, but after the model was trained, it can be encouraged to behave less erratically with an energy penalty of 0.01

All other arguments were set to their default values

# Plot of Rewards
[//]: # (Image References)

[image1]: https://github.com/hoomic/continuous-control-drlnd/blob/master/plot_of_rewards.png "Plot of Rewards"
![Plot of Rewards][image1]

# Ideas for Future Work

* During training, I noticed that the agent didn't seem to learn the rotational symmetry of the problem, and as a result it seemed to learn how to touch the object in some regions quickly and not others. For example, it might first touch the object when it is towards its south-west, then in future episodes, it seems to be able to find the object when it is in the south-west better than any other direction. One idea to alleviate this is to have it sample more experiences from areas in the state space where its reward is relatively low. That way, it isn't repeatedly sampling experiences where it is already doing well. I believe that prioritized experience replay would alleviate this problem somewhat, but I did not implement it in this project. So, one idea is to implement prioritized replay. Another idea is to use a clustering algorithm like KMeans to cluster the experiences by their state spaces into K clusters. Each cluster would have a cumulative reward for all the experiences in that cluster. You could then sample more often from clusters with a lesser cumulative reward than others, and the sampling from within a cluster would be uniform.
* Implement Trust Region Policy Optimization
* Implement DDPG using the multiple, parallel agent environment
* Add a small reward based on the distance between the reacher and the object that can be applied at every timestep
