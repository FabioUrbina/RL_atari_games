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

device = torch.device("cpu")



class Logger:
    
    def __init__(self, filename, model_name):
        self.filename = filename
        self.modelname = model_name
        f = open(f"{self.filename}.csv", "w")
        f.close()
        
    def log(self, msg):
        f = open(f"{self.filename}.csv", "a+")
        f.write(f"{msg}\n")
        f.close()
    

class Env_Runner:
    
    def __init__(self, env, agent, logger_folder):
        super().__init__()
        
        self.env = env
        self.agent = agent

        self.logger = Logger(f'{logger_folder}/training_info', agent.name)
        if not os.path.isfile(f'{logger_folder}/training_info.csv'):
            self.logger.log("training_step, return, model")
        
        self.ob = self.env.reset()
        
    def run(self, steps):
        
        global cur_step
        
        obs = []
        actions = []
        rewards = []
        dones = []
        values = []
        action_prob = []
        
        for step in range(steps):
            
            self.ob = torch.tensor(self.ob).to(device).to(dtype)
            policy, value = self.agent(self.ob.unsqueeze(0))
            action = self.agent.select_action(policy.detach().cpu().numpy()[0])
            
            obs.append(self.ob)
            actions.append(action)
            values.append(value.detach())
            action_prob.append(policy[0,action].detach())
            
            self.ob, r, done, info, additional_done = self.env.step(action)
            #what = self.env.step(action)
            #print(what)
            if done: # real environment reset, other add_dones are for learning purposes
                self.ob = self.env.reset()
                if "return" in info:
                    self.logger.log(f'{cur_step+step},{info["return"]},{self.agent.name}')
            
            rewards.append(r)
            dones.append(done or additional_done)
            
        cur_step += steps
                                    
        return [obs, actions, rewards, dones, values, action_prob]


class Batch_DataSet(torch.utils.data.Dataset):

    def __init__(self, obs, actions, adv, v_t, old_action_prob):
        super().__init__()
        self.obs = obs
        self.actions = actions
        self.adv = adv
        self.v_t = v_t
        self.old_action_prob = old_action_prob
        
    def __len__(self):
        return self.obs.shape[0]
    
    def __getitem__(self, i):
        return self.obs[i],self.actions[i],self.adv[i],self.v_t[i],self.old_action_prob[i]

class PPO_Network(nn.Module):
    # nature paper architecture
    
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.num_actions = num_actions
        
        network = [
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,512),
            nn.ReLU(),
            nn.Linear(512, num_actions + 1)
        ]
        
        self.network = nn.Sequential(*network)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        policy, value = torch.split(self.network(x),(self.num_actions, 1), dim=1)
        policy = self.softmax(policy)
        return policy, value


class PPO_Agent(nn.Module):
    
    def __init__(self, in_channels, num_actions, name=''):
        super().__init__()
        
        self.name = name
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.network = PPO_Network(in_channels, num_actions)
    
    def forward(self, x):
        policy, value = self.network(x)
        return policy, value
    
    def select_action(self, policy):
        return np.random.choice(range(self.num_actions) , 1, p=policy)[0]


class Atari_Wrapper(gym.Wrapper):
    # env wrapper to resize images, grey scale and frame stacking and other misc.
    
    def __init__(self, env, env_name, k, dsize=(84,84), use_add_done=False):
        super(Atari_Wrapper, self).__init__(env)
        self.dsize = dsize
        self.k = k
        self.use_add_done = use_add_done
        
        # set image cutout depending on game
        if "Pong" in env_name:
            self.frame_cutout_h = (33,-15)
            self.frame_cutout_w = (0,-1)
        elif "Breakout" in env_name:
            self.frame_cutout_h = (31,-16)
            self.frame_cutout_w = (7,-7)
        elif "SpaceInvaders" in env_name:
            self.frame_cutout_h = (25,-7)
            self.frame_cutout_w = (7,-7)
        elif "Seaquest" in env_name:
            self.frame_cutout_h = (30,-30)
            self.frame_cutout_w = (9,-3)
        else:
            # no cutout
            self.frame_cutout_h = (0,-1)
            self.frame_cutout_w = (0,-1)
        
    def reset(self):
    
        self.Return = 0
        self.last_life_count = 0
        
        ob, __ = self.env.reset()
        ob = self.preprocess_observation(ob)
        
        # stack k times the reset ob
        self.frame_stack = np.stack([ob for i in range(self.k)])
        
        return self.frame_stack
    
    
    def step(self, action): 
        # do k frameskips, same action for every intermediate frame
        # stacking k frames
        
        reward = 0
        done = False
        additional_done = False
        
        # k frame skips or end of episode
        frames = []
        for i in range(self.k):
            
            ob, r, d1, d2, info = self.env.step(action)
            d = 0
            if d1 or d2:
                d = 1
            # insert a (additional) done, when agent loses a life (Games with lives)
            if self.use_add_done:
                if info['lives'] < self.last_life_count:
                    additional_done = True  
                self.last_life_count = info['lives']
            
            ob = self.preprocess_observation(ob)
            frames.append(ob)
            
            # add reward
            reward += r
            
            if d: # env done
                done = True
                break
                       
        # build the observation
        self.step_frame_stack(frames)
        
        # add info, get return of the completed episode
        self.Return += reward
        if done:
            info["return"] = self.Return
            
        # clip reward
        if reward > 0:
            reward = 1
        elif reward == 0:
            reward = 0
        else:
            reward = -1
            
        return self.frame_stack, reward, done, info, additional_done
    
    def step_frame_stack(self, frames):
        
        num_frames = len(frames)
        
        if num_frames == self.k:
            self.frame_stack = np.stack(frames)
        elif num_frames > self.k:
            self.frame_stack = np.array(frames[-self.k::])
        else: # mostly used when episode ends 
            
            # shift the existing frames in the framestack to the front=0 (0->k, index is time)
            self.frame_stack[0: self.k - num_frames] = self.frame_stack[num_frames::]
            # insert the new frames into the stack
            self.frame_stack[self.k - num_frames::] = np.array(frames)  
            
    def preprocess_observation(self, ob):
    # resize and grey and cutout image
    
        ob = cv2.cvtColor(ob[self.frame_cutout_h[0]:self.frame_cutout_h[1],
                           self.frame_cutout_w[0]:self.frame_cutout_w[1]], cv2.COLOR_BGR2GRAY)
        ob = cv2.resize(ob, dsize=self.dsize)
    
        return ob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

def compute_advantage_and_value_targets(rewards, values, dones, gamma, lam):
    
    advantage_values = []
    old_adv_t = torch.tensor(0.0).to(device)
    
    value_targets = []
    old_value_target = values[-1]
    
    for t in reversed(range(len(rewards)-1)):
        
        if dones[t]:
            old_adv_t = torch.tensor(0.0).to(device)
        
        # ADV
        delta_t = rewards[t] + (gamma*(values[t+1])*int(not dones[t+1])) - values[t]
        
        A_t = delta_t + gamma*lam*old_adv_t
        advantage_values.append(A_t[0])
        
        old_adv_t = delta_t + gamma*lam*old_adv_t
        
        # VALUE TARGET
        value_target = rewards[t] + gamma*old_value_target*int(not dones[t+1])
        value_targets.append(value_target[0])
        
        old_value_target = value_target
    
    advantage_values.reverse()
    value_targets.reverse()
    
    return advantage_values, value_targets


def train(args):  
    
    # create folder to save networks, csv, hyperparameter
    folder_name = time.asctime(time.gmtime()).replace(" ","_").replace(":","_")
    folder_name = 'Centipede'
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    
    # if checkpoint exists, load the checkpoint
   
    # save the hyperparameters in a file
    f = open(f'{folder_name}/args.txt','w')
    for i in args.__dict__:
        f.write(f'{i},{args.__dict__[i]}\n')
    f.close()
    
    # arguments
    env_name = args.env
    num_stacked_frames = args.stacked_frames
    start_lr = args.lr 
    gamma = args.gamma
    lam = args.lam
    minibatch_size = args.minibatch_size
    T = args.T
    c1 = args.c1
    c2 = args.c2
    actors = args.actors
    start_eps = args.eps
    epochs = args.epochs
    total_steps = args.total_steps
    save_model_steps = args.save_model_steps
    savename = args.name
    checkpoint = args.checkpoint
    checkpoint_path = args.checkpoint_path
    # init
    
    # in/output    
    in_channels = num_stacked_frames
    num_actions = gym.make(env_name).env.action_space.n

    # network and optim
    if checkpoint:
        agent = torch.load(checkpoint_path)
    else:
        agent = PPO_Agent(in_channels, num_actions, name=savename).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=start_lr)
    
    # actors
    env_runners = []
    for actor in range(actors):

        raw_env = gym.make(env_name)
        env = Atari_Wrapper(raw_env, env_name, num_stacked_frames, use_add_done=args.lives)
        
        env_runners.append(Env_Runner(env, agent, folder_name))
        
    num_model_updates = 0

    #for n in tqdm(range(1000)):
    progress = tqdm(total=total_steps)
    while cur_step < total_steps:
        
        # change lr and eps over time
        alpha = 1 - (cur_step / total_steps)
        current_lr = start_lr * alpha
        current_eps = start_eps * alpha
        
        #set lr
        for g in optimizer.param_groups:
            g['lr'] = current_lr
        
        # get data
        batch_obs, batch_actions, batch_adv, batch_v_t, batch_old_action_prob = None, None, None, None, None
        # random reward of a random trajectory to plot
        random_rewards = []
        for env_runner in env_runners:
            obs, actions, rewards, dones, values, old_action_prob = env_runner.run(T)
            adv, v_t = compute_advantage_and_value_targets(rewards, values, dones, gamma, lam)

            # get the rewards added
            random_rewards.append(sum(rewards))
            # assemble data from the different runners 
            batch_obs = torch.stack(obs[:-1]) if batch_obs == None else torch.cat([batch_obs,torch.stack(obs[:-1])])
            batch_actions = np.stack(actions[:-1]) if batch_actions is None else np.concatenate([batch_actions,np.stack(actions[:-1])])
            batch_adv = torch.stack(adv) if batch_adv == None else torch.cat([batch_adv,torch.stack(adv)])
            batch_v_t = torch.stack(v_t) if batch_v_t == None else torch.cat([batch_v_t,torch.stack(v_t)]) 
            batch_old_action_prob = torch.stack(old_action_prob[:-1]) if batch_old_action_prob == None else torch.cat([batch_old_action_prob,torch.stack(old_action_prob[:-1])])
            progress.update(T)

        # load into dataset/loader
        dataset = Batch_DataSet(batch_obs,batch_actions,batch_adv,batch_v_t,batch_old_action_prob)
        dataloader = DataLoader(dataset, batch_size=minibatch_size, num_workers=0, shuffle=True)
        
        
        # update
        for epoch in range(epochs):
             
            # sample minibatches
            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                if i >= 8:
                    break
                
                # get data
                obs, actions, adv, v_target, old_action_prob = batch 
                
                adv = adv.squeeze(1)
                # normalize adv values
                adv = ( adv - torch.mean(adv) ) / ( torch.std(adv) + 1e-8)
                
                # get policy actions probs for prob ratio & value prediction
                policy, v = agent(obs)
                # get the correct policy actions
                pi = policy[range(minibatch_size),actions.long()]
                
                # probaility ratio r_t(theta)
                probability_ratio = pi / (old_action_prob + 1e-8)
                
                # compute CPI
                CPI = probability_ratio * adv
                # compute clip*A_t
                clip = torch.clamp(probability_ratio,1-current_eps,1+current_eps) * adv     
                
                # policy loss | take minimum
                L_CLIP = torch.mean(torch.min(CPI, clip))
                
                # value loss | mse
                L_VF = torch.mean(torch.pow(v - v_target,2))
                
                # policy entropy loss 
                S = torch.mean( - torch.sum(policy * torch.log(policy + 1e-8),dim=1))

                loss = - L_CLIP + c1 * L_VF - c2 * S
                loss.backward()
                optimizer.step()
        
            
        num_model_updates += 1
         
        
        # save the network after some time
        if cur_step%save_model_steps < T*actors:
            torch.save(agent,f'{folder_name}/{savename}-{cur_step}.pt')

    torch.save(agent,f'{folder_name}/{savename}-FINAL.pt')
    progress.close()
    env.close()
if __name__ == "__main__":
    cur_step = 0
    args = argparse.ArgumentParser()
    
    # set hyperparameter
    args.add_argument('-lr', type=float, default=2.5e-4)
    args.add_argument('-env', default='ALE/Centipede-v5')
    args.add_argument('-name', default='Centipede-v5')  
    args.add_argument('-lives', type=bool, default=True)
    args.add_argument('-stacked_frames', type=int, default=4)
    args.add_argument('-gamma', type=float, default=0.99)
    args.add_argument('-lam', type=float, default=0.95)
    args.add_argument('-eps', type=float, default=0.2)
    args.add_argument('-c1', type=float, default=.95)
    args.add_argument('-c2', type=float, default=0.01)
    args.add_argument('-minibatch_size', type=int, default=64)
    args.add_argument('-actors', type=int, default=4)
    args.add_argument('-T', type=int, default=256)
    args.add_argument('-epochs', type=int, default=3)
    args.add_argument('-total_steps', type=int, default=1000000)
    args.add_argument('-save_model_steps', type=int, default=200000)
    args.add_argument('-checkpoint', type=bool, default=False)
    args.add_argument('-checkpoint_path', type=str, default='')
    args.add_argument('-report', type=int, default=10)
    
    train(args.parse_args())