import torch
import torch.nn as nn
import numpy as np

def convLayer(in_channels,out_channels,kernel_size = 3,stride = 1,padding = 1):
    #default params maintain dimensions
    conv = nn.Sequential(nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size = kernel_size,
                                   stride = stride,
                                   padding = padding),
                         nn.LeakyReLU())
    return conv

def fcLayer(in_channels,out_channels):
    fc = nn.Sequential(nn.Linear(in_channels,out_channels),
                       nn.LeakyReLU())
    return fc

class AtariNetCONV(nn.Module):
    #The Atari games have a frame size of 210x160, but we can crop it for some of them
    def __init__(self, inshape = [210,160], poolsizes = [2,2], numconvlayers = 3, outsize = 1):
        super(AtariNetCONV, self).__init__()
        self.inshape = inshape
        self.insize = np.prod(inshape)
        self.outsize = outsize
            
        #input layer 210 x 160
        self.layers = [nn.MaxPool2d(kernel_size = poolsizes[0], stride = poolsizes[0])]
        
        #conv layers
        self.layers.append(convLayer(1,10))
        for i in range(numconvlayers-1):
            self.layers.append(convLayer(10,10))

        self.layers.append(nn.MaxPool2d(kernel_size = poolsizes[1], stride = poolsizes[1]))
        
        height = inshape[0]//poolsizes[0]//poolsizes[1]
        width =  inshape[1]//poolsizes[0]//poolsizes[1]

        self.layers.append(fcLayer(10*height*width,500))
        self.layers.append(nn.Linear(500,outsize))
        self.numlayers = len(self.layers)

    def forward(self,x):
        out = self.layers[0](x)
        for i in range(1,self.numlayers-2):
            out = self.layers[i](out)
        out = out.reshape(out.size(0),-1)
        out = self.layers[-2](out)
        out = self.layers[-1](out)
            
        if self.outsize != 1:
            out = nn.Softmax(dim = 1)(out)
            out = out.flatten()
        return out
    
    def mutate(self,mut_power,mutations):
        theta = self.state_dict()
        for seed in mutations:
            for i,name in enumerate(theta):
                weights = theta[name]
                torch.manual_seed(seed+i) #seed+i for each layer is still sampling from N,
                                          #it's just easier to do it for each layer individually
                weights += mut_power * torch.empty(weights.shape).normal_(mean=0,std=1)
                    
    def addlayer(self,size):
        pass
    
    def save(self,filename,optimizer,measures = None):
        state = {
            'measures': measures,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filename)
        
    def load(self,filename, optimizer, measures = None):
        checkpoint = torch.load(filename, map_location = 'cpu')
        measures = checkpoint['measures']
        self.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return measures, optimizer 

class AtariNetFC(nn.Module):
    #The Atari games have a frame size of 210x160, but we can crop it for some of them
    def __init__(self, inshape = [210,160], midsize = 500, outsize = 1):
        super(AtariNetFC, self).__init__()
        self.insize = np.prod(insize)
        self.midsize = midsize
        self.outsize = outsize
            
        #input layer 210 x 160
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        #new size 105 x 80
        self.l1 = nn.Sequential(
                    nn.Linear(self.insize//4,self.midsize),
                    nn.LeakyReLU())
        
        self.lout = nn.Linear(self.midsize,self.outsize)
        self.layers = [self.pool1,self.l1,self.lout]
        self.numlayers = len(self.layers)

    def forward(self,x):
        out = self.layers[0](x)
        out = out.reshape(out.size(0),-1)
        for i in range(1,self.numlayers):
            out = self.layers[i](out)
            
        if self.outsize != 1:
            out = nn.Softmax(dim = 1)(out)
            out = out.flatten()
        return out
    
    def mutate(self,mut_power,mutations):
        theta = self.state_dict()
        for seed in mutations:
            for i,name in enumerate(theta):
                weights = theta[name]
                torch.manual_seed(seed+i) #seed+i for each layer is still sampling from N,
                                          #it's just easier to do it for each layer individually
                weights += mut_power * torch.empty(weights.shape).normal_(mean=0,std=1)
                    
    def addlayer(self,size):
        pass
    
    def save(self,filename,optimizer,measures = None):
        state = {
            'measures': measures,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filename)
        
    def load(self,filename, optimizer, measures = None):
        checkpoint = torch.load(filename, map_location = 'cpu')
        measures = checkpoint['measures']
        self.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return measures, optimizer 
    
def CustomLoss(coef,logprob):
    return -torch.mean(coef * logprob) #- because we want to travel in negative of negative gradient
        
    
def ProcessIm(game,t,obs,prev_obs):
    if game == 'Tetris':
        #Isolate just the tetris game part
        #it shakes up and down every other timestep
         #even timestep
        if t%2 == 0:
            obs1 = obs[27:203,22:64]
            obs2 = prev_obs[28:204,22:64]
        #odd timestep
        else:
            obs1 = obs[28:204,22:64]
            obs2 = prev_obs[27:203,22:64]
   
    elif game == 'DemonAttack':
        obs1 = obs
        obs2 = prev_obs
    observation = torch.Tensor(obs1 - obs2)
    observation.unsqueeze_(0)
    observation.unsqueeze_(0)
    return observation
        
