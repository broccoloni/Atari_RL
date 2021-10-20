#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gym
import time
import matplotlib.pyplot as plt
import torch
from AtariNet import *
from tqdm import tqdm
from ale_py import ALEInterface
ale = ALEInterface()

almightyinty = 6671111 #call pizza pizza


# In[2]:


env = gym.make('ALE/DemonAttack-v5', render_mode='rgb_array')
env = gym.wrappers.GrayScaleObservation(env)


# In[3]:


#hyperparameters
gamma = 0.99
atarilr = 0.001
criticlr = 0.001
num_batches = 1000
cur_batches = 800
#10    3000
#100   2000
#1000  1000
batch_size = 20
numactions = 6
maxframes = 2000
insize = [210,160,1]

midsize = 100

#initializations
atarinet = AtariNet(insize = insize, midsize = midsize, outsize = numactions)
critic = AtariNet(insize = insize, midsize = midsize, outsize = 1)
atariOptimizer = torch.optim.Adam(atarinet.parameters(), lr = atarilr, weight_decay = 0)
criticOptimizer = torch.optim.Adam(critic.parameters(), lr = criticlr, weight_decay = 0)


# In[ ]

sum_rewards = []
sum_rewards_est = []
batch_lens = []

if cur_batches != 0:
    measures, atariOptimizer = atarinet.load("DemonAttack_m{}_{}.tar".format(midsize,cur_batches),atariOptimizer)
    measures, criticOptimizer = critic.load("Critic_DemonAttack_m{}_{}.tar".format(midsize,cur_batches),criticOptimizer)

    sum_rewards = measures[0]
    sum_rewards_est = measures[1]
    episode_lens = measures[2]
    cur_batches = len(sum_rewards)

for i_batch in tqdm(range(cur_batches,cur_batches+num_batches)):
    batch_logprobs = []
    batch_rewards = []
    batch_rewards_est = []
    batch_ep_lens = []

    for i_episode in range(batch_size):
        ep_rewards = 0  #in case of sparse rewards, like tetris
        while ep_rewards == 0:
            logprobs = []
            rewards = []
            rewards_est = []
            obs = env.reset()
            prev_obs = np.zeros(obs.shape)
            
            cur_lives = 4
            prev_lives = 4
            for t in range(maxframes):
    #             env.render(mode='human')

                #Preprocessing
                observation = ProcessIm_DemonAttack(obs,prev_obs)    

                #Forward pass
                probs = atarinet.forward(observation)
                reward_est = critic.forward(observation)

                #Sample action space from probability distribution
                action = np.random.choice(numactions, 1, p = probs.detach().numpy())[0]

                #Take action
                prev_obs = obs
                obs, reward, done, info = env.step(action)
                
                #Add negative reward for losing a life
                cur_lives = info['lives']
                if t != 0 and cur_lives != prev_lives:
                    reward -= 50
                prev_lives = cur_lives

                #break the sparse rewards loop
                if reward != 0:
                    ep_rewards = 1

                #Store values of episode
                logprobs.append(torch.log(probs[action]))
                rewards.append(reward)
                rewards_est.append(reward_est)

                if done:
                    batch_ep_lens.append(t)
    #                 print("Episode finished after {} timesteps".format(t+1))
                    break
        
            batch_logprobs.append(logprobs)
            batch_rewards.append(rewards)
            batch_rewards_est.append(rewards_est)
            batch_ep_lens.append(t)

    #Backprop - One step actor critic method from Sutton & Barto book
    #just do batch size 1 right now, implement batches if I have time
    batch_len = np.min(batch_ep_lens)
    batch_logprobs = torch.Tensor([ep[:batch_len] for ep in batch_rewards]).requires_grad_()
    batch_rewards = torch.Tensor([ep[:batch_len] for ep in batch_rewards]).requires_grad_()
    batch_rewards_est = torch.Tensor([ep[:batch_len] for ep in batch_rewards_est]).requires_grad_()

    discounts = torch.ones(batch_size,batch_len)*gamma
    powers = torch.arange(0,batch_len)
    gammas = torch.pow(discounts,powers).requires_grad_()
    for t in range(batch_len-1):        
        G = torch.sum(gammas[:,:batch_len - t - 1] * batch_rewards[:,t+1:],axis = 1)
        g = gammas[:,t]
        delta = G - batch_rewards_est[:,t]
        
        criticloss = CustomLoss(delta,batch_rewards_est[:,t].reshape(batch_size,1))
        criticloss.backward(retain_graph = True)
        
        atariloss = CustomLoss(g*delta,batch_logprobs[:,t].reshape(batch_size,1))
        atariloss.backward(retain_graph = True)
        
    sum_rewards.append(np.sum(rewards))
    sum_rewards_est.append(torch.sum(batch_rewards_est).detach())
    batch_lens.append(batch_len)

    if i_batch != cur_batches and i_batch%100 == 0:
        measures = [sum_rewards,sum_rewards_est,batch_lens]
        atarinet.save("DemonAttack_m{}_{}.tar".format(midsize,i_batch),atariOptimizer,measures = measures)
        critic.save("Critic_DemonAttack_m{}_{}.tar".format(midsize,i_batch),criticOptimizer, measures = measures)

env.close()

#Part of original numpy implementation which was too slow:
#
## In[5]:
#
#plt.plot(sum_rewards)
#plt.show()
#plt.plot(sum_rewards_est)
#plt.show()
#plt.plot(episode_lens)
#plt.show()
#
#
## In[24]:
#
#
#measures = [sum_rewards,sum_rewards_est,episode_lens]
#tetrisnet.save("TetrisNet.tar",tetrisOptimizer,measures = measures)
#critic.save("Critic.tar",criticOptimizer,measures = measures)
#
#
## In[4]:
#
#
#tetrislr = 0.001
#criticlr = 0.001
#tetrisnet = TetrisNet()
#critic = TetrisNet(outsize = 1)
#
#tetrisOptimizer = torch.optim.Adam(tetrisnet.parameters(), lr = tetrislr, weight_decay = 0)
#criticOptimizer = torch.optim.Adam(tetrisnet.parameters(), lr = criticlr, weight_decay = 0)
#
#measures, tetrisOptimizer = tetrisnet.load("TetrisNet.tar",tetrisOptimizer)
#measures, criticOptimizer = critic.load("Critic.tar",criticOptimizer)
#
#sum_rewards = measures[0]
#sum_rewards_est = measures[1]
#episode_lens = measures[2]
#
#
## In[5]:
#
#
##Watch it play
#
#observation = env.reset()
#for t in range(10000):
#    env.render(mode='human')
#    observation = ProcessIm(t,observation)
#
#    probs = tetrisnet.forward(observation)
#    action = np.random.choice(18, 1, p = probs.detach().numpy())[0]
#    observation, reward, done, info = env.step(action)
#
#
## In[5]:
#
#
#tetrisnet = TetrisNet()
#print(tetrisnet.state_dict().keys())
#print(tetrisnet.state_dict()['conv1.0.weight'].shape)
#print(tetrisnet.state_dict()['conv1.0.bias'].shape)
#print(tetrisnet.state_dict()['conv1.1.weight'].shape)
#print(tetrisnet.state_dict()[''])
#
#
## In[ ]:
#
#
#from gym import envs
#print(envs.registry.all())
#
#
## In[ ]:
#
#
#def MaxPool(arr,shape = [2,2]):
#    height, width = shape
#    dims = len(arr.shape)
#    if dims == 3:
#        xdim,ydim,c = arr.shape
#
#    else:
#        print("Error: I only made maxpool for 3d tensors")
#        return
#
#    xdim_new = xdim//width
#    ydim_new = ydim//height
#    
#    arr_new = np.max(arr.reshape((xdim_new, width, ydim_new, height,c)),axis = (1,3))
#    return arr_new
#
#
#
#def ReLU(x):
#    x[x<0] = 0
#    return x
#    
#def BatchNorm():
#    pass
#    
#class TetrisNet():
#    def __init__(self, weights = None,lr = 0.001):
#        self.name = "TetrisNet"
#        self.l1nodes    = 1000
#        self.outnodes   = 18
#        self.lr = lr
#        
#        divisor = 100   #outputs are very large at the beginning
#                        #adjusted this until outputs from forward were reasonably different
#        
#        if weights is None:
#            self.l1weights  = (2 * np.random.random((105 * 80 * 3, self.l1nodes)) - 1)/divisor
#            self.l1biases   = (2 * np.random.random((self.l1nodes)) - 1)/divisor
#
#
#            self.outweights = (2 * np.random.random((self.l1nodes, self.outnodes)) -1)/divisor
#            self.outbiases  = (2 * np.random.random((self.outnodes)) - 1)/divisor
#
#        else:
#            self.l1weights = weights[0]
#            self.l1biases = weights[1]
#            self.outweights = weights[2]
#            self.outbiases = weights[3]
#        
#        self.l1 = [self.l1weights, self.l1biases]
#
#        self.out = [self.outweights, self.outbiases]
#
#        self.allweights = [self.l1weights,
#                           self.l1biases,
#                           self.outweights,
#                           self.outbiases]
#        
#        self.l1outputs = None
#        self.sumexp = -1
#        self.output = None
#        
#    def forward(self,x):
#        out = self.apply(MaxPool,  x)
#        out = out.flatten()
#        out = self.apply(self.l1,  out)
#        out = self.apply(ReLU,     out)
#        self.l1outputs = out
#        out = self.apply(self.out, out)
#        self.output = out
#        return out
#    
#    def predict(self,x):
#        return np.argmax(self.SoftMax(self.forward(x)))
#    
#    def backward(self,G,a,gamma):
#        onehota = np.zeros(self.outnodes)
#        onehota[a] = 1
#        update = np.outer(self.l1outputs, onehota) 
#
#        
#        update = self.l1outputs.reshape(-1,1) * self.outweights
#    
#    def apply(self,f,x,fparams = None):
#        #Check if it's a list (being the weights and biases of a layer)
#        if isinstance(f,list):
#            out = f[0].T.dot(x) + f[1]
#            return out
#        
#        #check if it's a function (ie, maxpool or relu)
#        elif callable(f):
#            if fparams is not None:
#                return f(x,fparams)
#            else:
#                return f(x)
#
#    def SumExp(self,x):
#        self.sumexp = np.sum(np.exp(x))
#                            
#    def SoftMax(self,x):
#        self.SumExp(x)
#        return np.exp(x)/self.sumexp
#    
#    def __str__(self):
#        print("Input\t\t\t\t->\t210 x 160 x 3")
#        print("MaxPool(2,2)\t210 x 160 x 3\t->\t105 x 80 x 3")
#        print("Linear\t\t\t{}\t->\t{}".format(25200,self.l1nodes))
#        print("Linear\t\t\t{}\t->\t{}".format(self.l1nodes,self.outnodes))
#        
#    def __repr__(self):
#        print("Input\t\t\t\t->\t210 x 160 x 3")
#        print("MaxPool(2,2)\t210 x 160 x 3\t->\t105 x 80 x 3")
#        print("Linear\t\t\t{}\t->\t{}".format(25200,self.l1nodes))
#        print("Linear\t\t\t{}\t->\t{}".format(self.l1nodes,self.outnodes))
#        
#    def save(self,fname):
#        np.savez(fname,self.allweights,allow_pickle = True)
#        
#    def load(self,fname):
#        data = np.load(fname,allow_pickle = True)
#        data = [data[key] for key in data]
#        data = data[1]
#        self.__init__(weights = data)
#
