import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import deque, namedtuple
import random
import numpy as np

Experience = namedtuple("Experience", field_names=[
                        "state", "action", "reward", "next_state", "done"])


class ReplayMemory(object):
  """
  Class adapted from PyTorch example:
  https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
  """

  def __init__(self, buffer_size, batch_size, seed):
    self.memory = deque(maxlen=buffer_size)
    self.batch_size = batch_size
    self.seed = random.seed(seed)

  def push(self, state, action, reward, next_state, done):
    self.memory.append(Experience(state, action, reward, next_state, done))

  def sample(self, device):
    """ 
    Sample a set memories.
    Code adapted from a post from Chanseok Kang:
    https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html
    """
    experiences = random.sample(self.memory, k=self.batch_size)
    #print(experiences)
    #for e in experiences:
      #print(e.state)
    states = torch.from_numpy(
        np.vstack([e.state for e in experiences if e is not None])).float().to(device)
    actions = torch.from_numpy(
        np.vstack([e.action for e in experiences if e is not None])).long().to(device)
    rewards = torch.from_numpy(
        np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack(
        [e.next_state for e in experiences if e is not None])).float().to(device)
    dones = torch.from_numpy(np.vstack(
        [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

    return (states, actions, rewards, next_states, dones)

  def __len__(self):
    return len(self.memory)


class DQN(nn.Module):
    

  def __init__(self, inputs, outputs):
    super().__init__()

    self.fc1 = nn.Linear(in_features=inputs, out_features=64)
    self.fc2 = nn.Linear(in_features=64, out_features=128)
    self.fc3 = nn.Linear(in_features=128, out_features=64)
    self.fc4 = nn.Linear(in_features=64, out_features=32)
    self.out = nn.Linear(in_features=32, out_features=outputs)

  def forward(self, t):
    t = F.relu(self.fc1(t))
    t = F.relu(self.fc2(t))
    t = F.relu(self.fc3(t))
    t = F.relu(self.fc4(t))
    t = self.out(t)
    return t

class DQNAgent():


  def __init__(
      self,
      state_vector_length,
      num_actions,
      alpha=.001,
      eps=1,
      eps_decay=0.995,
      eps_min=0.05,
      gamma=0.9,
      batch_size=64,
      seed=None
  ):
    self.num_actions = num_actions
    self.eps = eps
    self.eps_decay = eps_decay
    self.eps_min = eps_min
    self.gamma = gamma
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.step = 0
    self.policy_net = DQN(state_vector_length, num_actions).to(self.device)
    self.target_net = DQN(state_vector_length, num_actions).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()
    self.optimizer = torch.optim.Adam(
        params=self.policy_net.parameters(), lr=alpha)

    self.memory = ReplayMemory(10000, batch_size, seed)

    if seed != None:
      np.random.seed(seed)

  def select_action(self, s):
    self.step += 1
    if np.random.random() < self.eps:
      action = np.random.randint(0, self.num_actions)
    else:
      action = self._get_best_action(s)

    return action

  def _get_best_action(self, s):
    with torch.no_grad():
      s = np.array(s)
      action = self.policy_net(torch.tensor([s]).to(
          self.device)).argmax(dim=1).to(self.device).item()
    return action

  def update_q(self, s, a, s_prime, r, done):
    self.memory.push(s, a, r, s_prime, done)
    self.step += 1

    if done:
      self.eps = max(self.eps_min, self.eps * self.eps_decay)

    if len(self.memory) > self.memory.batch_size:
      experiences = self.memory.sample(self.device)
      self.double_q_learn(experiences)

  def double_q_learn(self, experiences):
    states, actions, rewards, next_states, dones = experiences
    q_selection = self.policy_net(next_states).detach().argmax(1)
    target_values = self.target_net(next_states).detach()
    next_q_values = target_values[torch.arange(len(target_values)), q_selection].unsqueeze(1)

    #next_q_values = self.target_net(
        #next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + self.gamma * next_q_values * (1 - dones)
    #print(f"q targets: {q_targets}")
    current_q_values = self.policy_net(states).gather(1, actions)
    #print(f"current q values: {current_q_values}")

    loss = F.mse_loss(current_q_values, q_targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()  

  def learn(self, experiences):
    states, actions, rewards, next_states, dones = experiences
    next_q_values = self.target_net(
        next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + self.gamma * next_q_values * (1 - dones)
    current_q_values = self.policy_net(states).gather(1, actions)

    loss = F.mse_loss(current_q_values, q_targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def update_target(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())

  def save_network(self, outfile):
    torch.save(self.policy_net.state_dict(), outfile)

  def load_network(self, infile):
    self.policy_net.load_state_dict(torch.load(infile))
    self.policy_net.eval()