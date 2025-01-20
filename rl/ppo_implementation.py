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
            env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env=env, video_folder=(f"./videos/{run_name}"), name_prefix="test-video", episode_trigger=lambda x: x % 1000 == 0)
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
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    return args

if  __name__ == '__main__':
    args = parse_args()
    print(args)


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'mps'
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
    print("envs.single_observation_space.shape", envs.single_observation_space.shape)
    print("envs.single_action_space.n", envs.single_action_space.n)

    agent = Agent(envs).to(device)
    print(agent)
    optimizer = optim.Adam(agent.parameters(), lr = args.learning_rate, eps = 1e-5)

    #ALGO LOGIC

    obs = torch.zeros((args.num_steps, args.num_envs)) + envs.single_observation_space.to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)) + envs.single_action_space.to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)


    GLOBAL_STEP = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.Tensor(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    print(num_updates)





