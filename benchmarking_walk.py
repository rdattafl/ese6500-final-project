# This file contains all the code for the TD3 and SAC RL algorithms across all benchmarking tasks in the DeepMind Control Suite.

# Path: benchmarking.py

### Imports

# Basic Libraries
from dm_control import suite, viewer
import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

# RL Algorithms
from td3_dm import td3
from core_td3_dm import MLPActorCritic as MLPActorCriticTD3

from sac_dm import sac
from core_sac_dm import MLPActorCritic as MLPActorCriticSAC


### Benchmarking
if __name__ == "__main__":
    # Initialize the hyperparameters for the benchmarking
    r0 = np.random.RandomState(42)

    # Environment function
    def env_fn():
        return suite.load('walker', 'walk', task_kwargs={'random': r0})
    
    e = env_fn()
    
    U=e.action_spec();udim=U.shape[0];
    X=e.observation_spec();xdim=14+1+9;

    ac_kwargs = {
        'hidden_sizes': [64, 64], 
        'activation': torch.nn.Tanh
    }
    
    # Define hyperparameters for TD3/SAC
    seed = 42
    steps_per_epoch = 1000
    epochs = 500
    gamma = 0.95 # gamma
    polyak = 0.99 # polyak
    clip_ratio = 0.2
    pi_lr = 1e-2
    vf_lr = 1e-4
    lam = 0.97
    max_ep_len = 1000

    # Run TD3 (using fixed set of hyperparameters shown above)
    # ac, average_test_returns, total_test_timesteps = td3(env_fn, actor_critic=MLPActorCriticTD3, ac_kwargs=ac_kwargs, seed=seed, 
    #     steps_per_epoch=steps_per_epoch, epochs=epochs, replay_size=int(1e6), gamma=gamma, 
    #     polyak=polyak, pi_lr=pi_lr, q_lr=vf_lr, batch_size=100, start_steps=10000, 
    #     update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
    #     noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000, 
    #     logger_kwargs=dict(), save_freq=1)

    # Run SAC (using fixed set of hyperparameters shown above)
    ac, average_test_returns, total_test_timesteps = sac(env_fn, actor_critic=MLPActorCriticSAC, ac_kwargs=ac_kwargs, seed=seed, 
        steps_per_epoch=steps_per_epoch, epochs=epochs, replay_size=int(1e6), gamma=gamma, 
        polyak=polyak, lr=1e-2, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1)

    # Store and plot the results
    plt.figure()
    plt.plot(total_test_timesteps, average_test_returns)
    plt.title(f"SAC Test Returns over All Timesteps")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Return")
    plt.legend(["Test Returns"])
    plt.savefig(f"SAC Test Returns over All Timesteps.png")

    # plt.title(f"TD3 Test Returns over All Timesteps")
    # plt.xlabel("Timesteps")
    # plt.ylabel("Average Return")
    # plt.legend(["Test Returns"])
    # plt.savefig(f"TD3 Test Returns over All Timesteps.png")

    # For use in the visualization
    vis_env = env_fn()

    # Define and use the policy function
    def policy(time_step):
        if time_step.last():
            return np.zeros(vis_env.action_spec().shape)
        obs = time_step.observation
        obs_array = np.array(obs['orientations'].tolist() + [obs['height']] + obs['velocity'].tolist())
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        action = ac.act(obs_tensor)
        return action.squeeze(0)
    
    # Launch the viewer
    viewer.launch(vis_env, policy=policy)



