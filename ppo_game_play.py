import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import cv2
import gymnasium as gym
from tqdm import tqdm
import argparse
import time
import os
from torch.utils.data import DataLoader
from PPO_game import Atari_Wrapper, PPO_Agent, Env_Runner, PPO_Network
device = torch.device("cpu")

dtype = torch.float
def run_agent(agent, ob):
               
        ob = torch.tensor(ob).to(device).to(dtype)
        policy, value = agent(ob.unsqueeze(0))
        action = agent.select_action(policy.detach().cpu().numpy()[0])
        
        return action

def play_game(env_name, agent, in_channels, lives):

    # in/output    
    raw_env = gym.make(env_name, render_mode="human")
    env = Atari_Wrapper(raw_env, env_name, in_channels, use_add_done=lives)
    
    # play ball

    observation = env.reset()

    for _ in range(10000):
        action = run_agent(agent, observation)
        observation, r, done, info, additional_done = env.step(action)
        if done:
            observation = env.reset()
if __name__ == "__main__":
    game = 'ALE/Centipede-v5'
    lives = True
    num_actions = gym.make(game).env.action_space.n
    agent = torch.load('/home/fabs/TheProjects/RL_atari_games/Centipede/Centipede-v5-FINAL.pt')
    play_game(game, agent, 4, lives)
    pass