# This file contains the implementation of the SAC off-policy RL Algorithm on which we will conduct benchmarking.

# Path: sac_dm.py

### Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import core_sac_dm as core
import itertools
from copy import deepcopy


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        indexes = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[indexes],
                     obs2=self.obs2_buf[indexes],
                     act=self.act_buf[indexes],
                     rew=self.rew_buf[indexes],
                     done=self.done_buf[indexes])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    

def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    spec = env.action_spec()

    # Get the dimensions for the specific task from the Deepmind Control Suite paper
    obs_dim = 14+1+9
    act_dim = spec.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = spec.maximum[0]

    # Create actor-critic module and target networks
    ac = actor_critic(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit, **ac_kwargs)
    ac_target = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_target.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_target.q1(o2, a2)
            q2_pi_targ = ac_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # # Useful info for logging
        # q_info = dict(Q1Vals=q1.detach().numpy(),
        #               Q2Vals=q2.detach().numpy())

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        return loss_pi

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_target.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    def test_agent():
        test_returns = []
        for _ in range(num_test_episodes):
            o, d, test_ep_ret, test_ep_len = test_env.reset(), False, 0, 0
            obs_test = o.observation
            obs_array_test = np.array(obs_test['orientations'].tolist() + [obs_test['height']] + obs_test['velocity'].tolist())
            while not(d or (test_ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                time_step_val_test = test_env.step(get_action(obs_array_test, True))
                obs_2_test = time_step_val_test.observation
                r_test = time_step_val_test.reward
                d = time_step_val_test.last()

                obs_array_test = np.array(obs_2_test['orientations'].tolist() + [obs_2_test['height']] + obs_2_test['velocity'].tolist())
                test_ep_ret += r_test
                test_ep_len += 1
            
            test_returns.append(test_ep_ret)
        
        test_returns = np.array(test_returns)
        return np.max(test_returns)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0
    obs = o.observation
    obs_array = np.array(obs['orientations'].tolist() + [obs['height']] + obs['velocity'].tolist())
    
    # For inside the for loop
    obs_arr_1 = obs_array

    # Initialize relevant arrays
    avg_test_returns = []
    total_test_timesteps = []
    cum_test_timesteps = 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        if t % 1000 == 0:
            print(f"Step: {t}")
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(obs_arr_1)
        else:
            a = np.random.uniform(spec.minimum, spec.maximum, spec.shape)

        # Step the env
        time_step_val = env.step(a)
        obs_2 = time_step_val.observation
        r = time_step_val.reward
        d = time_step_val.last()
    
        obs_arr_2 = np.array(obs_2['orientations'].tolist() + [obs_2['height']] + obs_2['velocity'].tolist())

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)y
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(obs_arr_1, a, r, obs_arr_2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        obs_arr_1 = obs_arr_2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling                   
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_return = test_agent()
            avg_test_returns.append(test_return)
            cum_test_timesteps += steps_per_epoch  # Update total test timesteps
            total_test_timesteps.append(cum_test_timesteps)  # Track the timestep at which the average was calculated
            print(f'Epoch: {epoch}/{epochs}')
            print(f'Test Return: {test_return}')
            print()


    print('Training complete')
    return ac, avg_test_returns, total_test_timesteps

