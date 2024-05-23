# This file contains the implementation of the TD3 off-policy RL Algorithm we will conduct benchmarking on.

# Path: sac_dm.py

### Imports
import numpy as np
import torch
from torch.optim import Adam
import core_td3_dm as core
import itertools


### Code

# Define the Replay Buffer class
class ReplayBuffer:
    def __init__(self, observation_dim, action_dim, size):
        self.observation_buffer = np.zeros(core.combined_shape(size, observation_dim), dtype=np.float32)
        self.observation2_buffer = np.zeros(core.combined_shape(size, observation_dim), dtype=np.float32)
        self.action_buffer = np.zeros(core.combined_shape(size, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.done_buffer = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, observation, action, reward, next_observation, done):
        self.observation_buffer[self.ptr] = observation
        self.observation2_buffer[self.ptr] = next_observation
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        indexes = np.random.randint(0, self.size, size=batch_size)
        batch = dict(observation=self.observation_buffer[indexes],
                     observation2=self.observation2_buffer[indexes],
                     action=self.action_buffer[indexes],
                     reward=self.reward_buffer[indexes],
                     done=self.done_buffer[indexes])
        
        # Return a dictionary of torch tensors
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
    

# Now, define the TD3 method; hyperparameters are the 5 mentioned below - 
# gamma (discount factor), polyak (interpolation factor in polyak averaging for target networks), 
# pi_lr (policy optimizer learning rate), q_lr (Q-network optimizer learning rate), and batch_size 
# (the mini-batch size for SGD).
def td3(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
    
    # Instantiate the environment
    env, test_env = env_fn(), env_fn()

    spec = env.action_spec()

    # Get the dimensions for the specific task from the Deepmind Control Suite paper
    obs_dim = 14+1+9
    act_dim = spec.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    action_limit = spec.maximum[0]

    # Create actor-critic module and target networks
    ac = actor_critic(obs_dim=obs_dim, act_dim=act_dim, act_limit=action_limit, **ac_kwargs)
    ac_target = actor_critic(obs_dim=obs_dim, act_dim=act_dim, act_limit=action_limit, **ac_kwargs)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_target.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(observation_dim=obs_dim, action_dim=act_dim, size=replay_size)

    # Loss function for updating Q-value networks
    def compute_q_loss(data):
        o, a, r, o2, d = data['observation'], data['action'], data['reward'], data['observation2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_target.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -action_limit, action_limit)

            # Target Q-values
            q1_pi_targ = ac_target.q1(o2, a2)
            q2_pi_targ = ac_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Loss function for updating policy
    def compute_pi_loss(data):
        o = data['observation']
        q1_pi = ac.q1(o, ac.pi(o))
        return -q1_pi.mean()
    
    # Separate optimizers for policy and Q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q = compute_q_loss(data)
        loss_q.backward()
        q_optimizer.step()

        # Possibly update pi and target networks
        if timer % policy_delay == 0:
            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_pi_loss(data)
            loss_pi.backward()
            pi_optimizer.step()
            
            # Unfreeze Q-networks so you can optimize it at next DDPG step
            for p in q_params:
                p.requires_grad = True
        
            # Finally, update target networks by polyak averaging
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -action_limit, action_limit)

    def test_agent():
        test_returns = []
        for _ in range(num_test_episodes):
            o_test, d, test_ep_ret, test_ep_len = test_env.reset(), False, 0, 0
            obs_test = o_test.observation
            obs_array_test = np.array(obs_test['orientations'].tolist() + [obs_test['height']] + obs_test['velocity'].tolist())
            while not(d or (test_ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                time_step_val_test = test_env.step(get_action(obs_array_test, 0))
                obs_2_test = time_step_val_test.observation
                r_test = time_step_val_test.reward
                d = time_step_val_test.last()
            
                obs_array_test = np.array(obs_2_test['orientations'].tolist() + [obs_2_test['height']] + obs_2_test['velocity'].tolist())
                test_ep_ret += r_test
                test_ep_len += 1
            
            test_returns.append(test_ep_ret)
        
        test_returns = np.array(test_returns)
        return np.max(test_returns)
    
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0
    obs = o.observation
    obs_array = np.array(obs['orientations'].tolist() + [obs['height']] + obs['velocity'].tolist())
    
    # For inside the for loop
    obs_arr_1 = obs_array

    # Initialize a list to store average returns and timesteps
    average_traj_returns = []
    average_test_returns = []

    total_traj_timesteps = []
    total_test_timesteps = []

    cum_traj_timesteps = 0
    cum_test_timesteps = 0

    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t % 1000 == 0:
            print(f"Step: {t}")

        if t > start_steps:
            a = get_action(obs_arr_1, act_noise)
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
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(obs_arr_1, a, r, obs_arr_2, d)

        # # Super critical, easy to overlook step: make sure to update 
        # # most recent observation!
        obs_arr_1 = obs_arr_2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            cum_traj_timesteps += ep_len  # Update total trajectory timesteps
            average_traj_returns.append(ep_ret)  # Track the average return
            total_traj_timesteps.append(cum_traj_timesteps)  # Track the timestep at which the average was calculated
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_return = test_agent()
            average_test_returns.append(test_return)
            cum_test_timesteps += steps_per_epoch  # Update total test timesteps
            total_test_timesteps.append(cum_test_timesteps)  # Track the timestep at which the average was calculated
            print(f'Epoch: {epoch}/{epochs}')
            print(f'Test Return: {test_return}')
            print()
            
        
    print('Training complete.')
    return ac, average_test_returns, total_test_timesteps