import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from tqdm import tqdm
device = torch.device("cpu")


class Grid:
    def __init__(self, n, m, exit_pos, figure_pos):
        self.n = n
        self.m = m
        self.exit_pos = np.array(exit_pos)
        self.figure_pos = np.array(figure_pos)

    def move(self, direction):
        x, y = self.figure_pos
        if direction == 0:
            if y < self.n-1:
                self.figure_pos[1] = self.figure_pos[1]+1
        elif direction == 1:
            if y > 0:
                self.figure_pos[1] = self.figure_pos[1]-1
        elif direction == 2:
            if x > 0:
                self.figure_pos[0] = self.figure_pos[0]-1
        elif direction == 3:
            if x < self.m-1:
                self.figure_pos[0] = self.figure_pos[0]+1

    def is_at_exit(self):

        return (self.figure_pos == self.exit_pos).all()

    def get_state(self, device):
        return torch.FloatTensor(self.figure_pos).unsqueeze(0).to(device)


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fca = nn.Linear(16, 32)
        self.fcb = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fca(x))
        x = nn.functional.relu(self.fcb(x))
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x


def generate_episode(grid, policy_net, device="cpu", max_episode_len = 100):
    state = grid.get_state(device)
    ep_length = 0
    while not grid.is_at_exit():
        # Convert state to tensor and pass through policy network to get action probabilities
        ep_length+=1
        action_probs = policy_net(state).squeeze()
        log_probs = torch.log(action_probs)
        cpu_action_probs = action_probs.detach().cpu().numpy()
        action = np.random.choice(np.arange(4), p=cpu_action_probs)


        # Take the action and get the new state and reward
        grid.move(action)
        next_state = grid.get_state(device)
        reward = -0.1 if not grid.is_at_exit() else 0

        # Add the state, action, and reward to the episode
        new_episode_sample = (state, action, reward)
        yield new_episode_sample, log_probs

        # We do not want to add the state, action, and reward for reaching the exit position
        if reward == 0:
            break

        # Update the current state
        state = next_state
        if ep_length > max_episode_len:
            return

    # Add the final state, action, and reward for reaching the exit position
    new_episode_sample = (grid.get_state(device), None, 0)
    yield new_episode_sample, log_probs


def gradients_wrt_params(
    net: torch.nn.Module, loss_tensor: torch.Tensor
):
    # Dictionary to store gradients for each parameter
    # Compute gradients with respect to each parameter
    for name, param in net.named_parameters():
        g = grad(loss_tensor, param, retain_graph=True)[0]
        param.grad = g

def update_params(net: torch.nn.Module, lr: float) -> None:
    # Update parameters for the network
    for name, param in net.named_parameters():
        param.data += lr * param.grad

"""
policy_net = PolicyNet()
policy_net.to(device)

lengths = []
rewards = []

gamma = 0.99
lr_policy_net = 2**-13
optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr_policy_net)

prefix = "reinforce-per-step"

for episode_num in tqdm(range(2500)):
    all_iterations = []
    all_log_probs = []    
    grid = Grid(5, 5, (5,0), (2,2))
    episode = list(generate_episode(grid, policy_net=policy_net, device=device))
    lengths.append(len(episode))
    loss = 0
    for t, ((state, action, reward), log_probs) in enumerate(episode[:-1]):
        gammas_vec = gamma ** (torch.arange(t+1, len(episode))-t-1)
        # Since the reward is -1 for all steps except the last, we can just sum the gammas
        G = - torch.sum(gammas_vec)
        rewards.append(G.item())
        policy_loss = log_probs[action]
        optimizer.zero_grad()
        gradients_wrt_params(policy_net, policy_loss)
        update_params(policy_net, lr_policy_net  * G * gamma**t)    
"""
if __name__ == "__main__":

    lengths = []
    rewards = []
    grid = Grid(5, 5, (5,0), (2,2))
    policy_net = PolicyNet()
    policy_net.to(device)


    gamma = 0.99
    lr_policy_net = 2**-11
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr_policy_net)
    for episode_num in tqdm(range(100)):
        all_iterations = []
        all_log_probs = []
        grid = Grid(5, 5, (3,3), (2,2))
        episode = list(generate_episode(grid, policy_net=policy_net, device=device))

        lengths.append(len(episode))
        loss = 0
        for t, ((state, action, reward), log_probs) in enumerate(episode[:-1]):
            gammas_vec = gamma ** (torch.arange(t+1, len(episode))-t-1)
            # Since the reward is -1 for all steps except the last, we can just sum the gammas
            G = - torch.sum(gammas_vec)

            rewards.append(G.item())
            policy_loss = log_probs[action]
            optimizer.zero_grad()
            gradients_wrt_params(policy_net, policy_loss)
            update_params(policy_net, lr_policy_net  * G * gamma**t) 

            print(sum(rewards)/len(rewards))
    
    grid = Grid(5, 5, (4,4), (2,2))
    episode = list(generate_episode(grid, policy_net=policy_net, device=device))

    print(episode)