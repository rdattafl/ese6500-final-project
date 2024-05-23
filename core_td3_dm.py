# This file contains the Actor, Q-Function Critic, and Actor-Critic modules for the TD3 off-policy RL algorithm.

# Path: core_td3_dm.py

### Imports
import numpy as np
import torch
import torch.nn as nn


### Helper Functions
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


### Actor and Critic Modules
class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, action_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.action_limit = action_limit

    def forward(self, observation):
        # Return the action output from the actor network (scaled based on action limit)
        return self.action_limit * self.pi(observation)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        q_sizes = [obs_dim + act_dim] + list(hidden_sizes) + [1]
        self.q = mlp(q_sizes, activation)

    def forward(self, observation, action):
        # Return the Q-value output from the critic network
        q_value = self.q(torch.cat([observation, action], dim=-1))
        return torch.squeeze(q_value, -1)
    

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, observation):
        with torch.no_grad():
            return self.pi(observation).numpy()




