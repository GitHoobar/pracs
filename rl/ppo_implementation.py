import argparse
import os
from distutils.util import strtobool
import random
import gym.wrappers.record_episode_statistics
import numpy as np
import gymnasium as gym 
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode = "rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env=env, video_folder=(f"./video_{run_name}"), name_prefix="test-video", episode_trigger=lambda x: x % 100 == 0)
        env.reset(seed=seed)
        return env
    return thunk

def layer_init(layer, std = np.sqrt(2), bias_control = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_control)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std = 1.),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std = 0.01),
        )
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action = None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                    help = "name of the experiment")
    parser.add_argument('--gym-id', type=str, default='CartPole-v1',
                    help = "the gym environment name")
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                    help = "lr for the optimizer")
    parser.add_argument('--seed', type = int, default=1,
                    help = "seed")
    parser.add_argument('--total-timesteps', type = int, default=25000,
                    help = "timesteps for the training")
    parser.add_argument('--torch-deterministic', type = lambda x:bool(strtobool(x)), default=True, nargs = '?', const=True,
                    help = "if toggeled -> torch.backends.cudnn.deterministic = False")
    parser.add_argument('--cuda',type= lambda x:bool(strtobool(x)), default=True, nargs='?', const = True,
                    help = "if toggled -> cuda will not be enabled by default")
    parser.add_argument('--capture-video', type = lambda x:bool(strtobool(x)), default= False, nargs='?', const=True,
                    help = "wether to captue video or not")
    
    #Algorithm arguments
    parser.add_argument('--num-envs', type = int, default=4,
                    help = "no of parallel games environment")
    parser.add_argument('--num-steps', type = int, default=128,
                    help = "no of steps to run in each environment per policy rollout")
    parser.add_argument('--anneal-lr', type = lambda x: bool(strtobool(x)), default = True, nargs='?', const = True,
                    help = "toggle learning rate annealing for policy and value network")
    parser.add_argument('--gae', type = lambda x: bool(strtobool(x)), default=True, nargs='?', const = True,
                    help = "use GAE for advantage computation")
    parser.add_argument('--gamma', type = float, default=0.99,
                    help = "discount factor")
    parser.add_argument('--gae-lambda', type = float, default=0.95,
                    help = "lambda for gae")
    parser.add_argument('--num-minibatches', type = int, default=4,
                    help = "no of minibatches")
    parser.add_argument('--update-epochs', type = int,default=4,
                    help = "the K epochs to update a policy ")
    parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default= True, nargs='?', const = True,
                    help = "toggles advantages informtion")
    parser.add_argument('--clip-coef', type=float, default=0.2,
                    help = "surrogate clipped objective coefficient")
    parser.add_argument('--clip-vloss', type = lambda x: bool(strtobool(x)), default=True, nargs='?', const = True,
                    help = "toggle whether to use a clipped loss or not")
    parser.add_argument('--ent-coef', type = float , default= 0.01,
                    help = "entropy loss coefficient")
    parser.add_argument('--vf-coef', type = float , default= 0.5,
                    help = "value function coefficient")
    parser.add_argument('--max-grad-norm', type = float , default= 0.5,
                    help = "the maximum norm of gradient clipping")
    parser.add_argument('--target-kl', type = float , default= None,
                    help = "target of kl divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__ == '__main__':
    args = parse_args()
    # print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    print(device)

    # env = gym.make("CartPole-v1", render_mode = 'rgb_array')
    # env = gym.wrappers.RecordVideo(env=env, video_folder="./videos", name_prefix="test-video", episode_trigger=lambda x: x % 1000 == 0)
    # env = gym.wrappers.RecordEpisodeStatistics(env)

    # observation = env.reset()
    # for _ in range(200):
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         observation = env.reset()
    #         print(f"episodic return {info['episode']['r']}")
    # env.close()

    #  NOTE: demo example
    # envs = gym.vector.SyncVectorEnv([make_env(args.gym_id)])
    # observation = envs.reset()
    # for _ in range (200):
    #     action = envs.action_space.sample()
    #     observation, reward, terminated, truncated, info = envs.step(action)
    #     for item in info:
    #         if "episode" in item:
    #             print(f"episodic return: {info['episode']['r']}")

    #env setup 
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, args.exp_name)
        for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # print("envs.single_observation_space.shape", envs.single_observation_space.shape)
    # print("envs.single_action_space.n", envs.single_action_space.n)

    agent = Agent(envs).to(device)
    # print(agent)
    optimizer = optim.Adam(agent.parameters(), lr = args.learning_rate, eps = 1e-5)

    #ALGO LOGIC
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    GLOBAL_STEP = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    # print(num_updates)
    # print("next_obs.shape", next_obs.shape)
    # print("agent.get_value(next_obs)", agent.get_value(next_obs))
    # print("agent.get_value(next_obs).shape", agent.get_value(next_obs).shape)
    # print()
    # print("agent.get_action_and_value(next_obs)", agent.get_action_and_value(next_obs))

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to:
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            GLOBAL_STEP += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob 

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(terminated).to(device)  

            for item in info:
                if "episode" in item:
                    print(f"global step = {GLOBAL_STEP}, episodic return = {info['episode']['r']}")
                    break
                      
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_value
                        next_values = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        next_values = values[t+1]
                    delta = rewards[t] + args.gamma * next_values * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        next_values = returns[t+1]
                    returns[t] = rewards[t] * args.gamma * nextnonterminal * next_return
                advantages = returns - values

            # flatten the batch 
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            #optimizing the policy and valuen network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    print("start and end index:", start, end)

                    b_obs_batch = b_obs[mb_inds].detach().requires_grad_(True)
                    b_actions_batch = b_actions.long()[mb_inds]
                    
                    # Remove any potential no_grad context
                    with torch.set_grad_enabled(True):
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                            b_obs_batch,  
                            b_actions_batch
                        )
                        
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()
                        
                        # Move the with torch.no_grad() block to only cover the KL computation
                        with torch.no_grad():
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                        
                        mb_advantages = b_advantages[mb_inds]
                        if args.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        
                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        
                        # Value loss
                        newvalue = newvalue.view(-1)
                        if args.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -args.clip_coef,
                                args.clip_coef
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                        
                        entropy_loss = entropy.mean()
                        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()

                    if args.target_kl is not None:
                        if approx_kl > args.target_kl:
                            break

                y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                
                print("SPS:", int(GLOBAL_STEP / (time.time() - start_time)), GLOBAL_STEP)
        
    envs.close()