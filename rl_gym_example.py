from models import DQNAgent
from tqdm import tqdm
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from collections import deque, namedtuple
import joblib
SEED = 42

def moving_average(data, window):
  series = pd.Series(data)
  return series.rolling(window).mean()

def plot_rewards(values):
  plt.figure(2)
  plt.clf()
  plt.xlabel('Episode')
  plt.ylabel('Reward')
  plt.plot(values)
  plt.plot(moving_average(values, 100))


def plot_multiple_rewards(variable, rewards_dict):
  plt.figure(2)
  plt.clf()
  plt.xlabel('Episode')
  plt.ylabel('Reward')
  for key, rewards in rewards_dict.items():
    plt.plot(rewards, label=f'{variable} = {key}')
  plt.legend()

def lander_runner(num_episodes, target_update, alpha, eps, eps_decay, gamma, seed, convergence_threshold=200, render=False):
  env = gym.make('LunarLander-v2')
  #env.seed(SEED)
  agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,
                   alpha=alpha, eps=eps, eps_decay=eps_decay, gamma=gamma, seed=SEED)

  rewards = []

  for e in tqdm(range(num_episodes)):
    cur_observation, info = env.reset()
    if render:
      env.render()
    episode_reward = 0
    for t in count():
      action = agent.select_action(cur_observation)
      next_observation, reward, term, truncated, info = env.step(action)
      done = 0
      if term or truncated == 1:
        done = 1
      agent.update_q(cur_observation, action, next_observation, reward, done)
      cur_observation = next_observation
      episode_reward += reward
      if render:
        env.render()
      if done:
        rewards.append(episode_reward)
        print(e)
        if e % 50 == 0:
          plot_rewards(rewards)
          plt.pause(0.01)
        #print(f'Episode {e}: {episode_reward}')
        break
    if e % target_update == 0:
      agent.update_target()
    if np.all(moving_average(rewards, 100)[-100:] >= convergence_threshold):
      print(f'Solved in {e} episodes.')
      agent.save_network(f'out\\agent.pt')
      break

  env.close()
  return rewards, agent


if __name__ == "__main__":

  run_rewards, agent = lander_runner(
  num_episodes=1000,
  target_update=8,
  alpha=0.0005,
  eps=1,
  eps_decay=0.99,
  gamma=0.999,
  seed=57,
  convergence_threshold=210
  )
  
  joblib.dump(agent, '/home/fabs/TheProjects/atari_models/doubleq_lunar_lander_bot_128.joblib')
  plot_rewards(run_rewards)

  env = gym.make("LunarLander-v2", render_mode="human")
  observation, info = env.reset()

  for _ in range(10000):
      action = agent.select_action(observation)  # agent policy that uses the observation and info
      #action = env.action_space.sample()
      observation, reward, terminated, truncated, info = env.step(action)
      if terminated or truncated:
          observation, info = env.reset()

  env.close()