#!/usr/bin/env python
# coding: utf-8
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
import torch
from AtariNet import *
from tqdm import tqdm
import os
from ale_py import ALEInterface

def evaluate(game,seed,mut,env,avg_rewards,i):
    #create initial model
    params = game_params(game)
    numactions = params[0]
    inshape = params[1]
    mut_power = params[2]
    test_size = params[3]
    poolsizes = params[4]
    numconvlayers = params[5]

    print("Making network")
    torch.manual_seed(seed)
    atarinet = AtariNetCONV(inshape=inshape,
                            poolsizes = poolsizes,
                            numconvlayers = numconvlayers, 
                            outsize = numactions).eval()

    print("mutating network")
    #apply mutations
    atarinet.mutate(mut_power,mut)
    print("beginning test")
    total_rewards = []
    for i_episode in range(test_size):
        #print(i,i_episode)
        obs = env.reset()
        prev_obs = np.zeros(obs.shape)
        rewards = []
        cur_lives = 4
        prev_lives = 4
        for t in range(5000):
#             env.render(mode='human')

            #Preprocessing
            observation = ProcessIm(game,t,obs,prev_obs)    

            #Forward pass
            probs = atarinet.forward(observation)

            #Sample action space from probability distribution
            action = np.random.choice(numactions, 1, p = probs.detach().numpy())[0]

            #Take action
            prev_obs = obs
            obs, reward, done, info = env.step(action)

            #Store values of episode
            rewards.append(reward)
        total_rewards.append(np.sum(rewards))
    #print("finished ind:",i)
    print("updating avg rewards")
    avg_rewards[i] = np.mean(total_rewards)
    print(avg_rewards)

def select_and_mutate(population,mutations,avg_rewards,pop_size,trunc):
    fit_order = np.argsort(avg_rewards)[::-1] #best to worst

    print("Average rewards:\t",np.mean(avg_rewards))
    print("Min reward:\t\t",avg_rewards[fit_order[-1]])
    print("Max reward:\t\t",avg_rewards[fit_order[0]])
    print() 

    #Probabilistic selection based off of rewards received
    #selection_probs = np.exp(avg_rewards[fit_order])/np.sum(np.exp(avg_rewards))
    #survivors = np.random.choice(pop_size+arch_size,pop_size,p = selection_probs)

    #cutoff selection
    survivors = fit_order[np.random.randint(0,trunc,size = pop_size)]
    arch_size = len(population) - pop_size
    survivors = np.append(survivors,fit_order[:arch_size])
    population = [population[i] for i in survivors]    

    mutations = [mutations[i] for i in survivors]
    for i in range(pop_size):
        mutations[i].append(np.random.randint(almightyint))

def load_gen(game,cur_gen,pop_size):
    data = np.load("{}/GAseeds_{}_nmp_{}.npz".format(game,game,cur_gen),allow_pickle = True)
    data = [data[key] for key in data]
    data = data[1]
    population = data[0]
    mutations = data[1]
    avg_rewards = np.array(data[2],dtype = float)
    select_and_mutate(population,mutations,avg_rewards,pop_size)
    return population,mutations,avg_rewards

def game_params(game):
    if game == 'Tetris':
        numactions = 18
        inshape = [176,42]
        mut_power = 0.002
        test_size = 3
        poolsizes = [3,2] #3 makes sense since tetris blocks are 3 pixels wide
        numconvlayers = 6 #makes field of vision of last layer full width of processed frame
        trunc = 3

    elif game == 'DemonAttack':
        numactions = 6
        inshape = [210,160]
        mut_power = 0.002
        test_size = 30
        poolsizes = [2,2]
        numconvlayers = 10
        trunc = 50

    return [numactions,inshape,mut_power,test_size,poolsizes,numconvlayers,trunc]

if __name__ == "__main__":
    game = "Tetris"
    almightyint = 6671111 #call pizza pizza hey hey hey
    pop_size = 5
    arch_size = 1
    num_gens = 2
    cur_gen = 0
    params = game_params(game)
    trunc = params[-1]

    ale = ALEInterface()
    env = gym.make('ALE/{}-v5'.format(game), render_mode='rgb_array')
    env = gym.wrappers.GrayScaleObservation(env)

    if cur_gen != 0:
        load_gen(game,cur_gen,pop_size)
    else:
        population = np.random.randint(0,almightyint,size = pop_size+arch_size)
        mutations = [[] for i in range(pop_size+arch_size)]
        avg_rewards = np.zeros(arch_size+pop_size)
    
    print("Starting GA:")
    for gen in range(cur_gen,cur_gen+num_gens):
        print("Generation:\t{}/{}".format(gen+1,num_gens))
        for i in tqdm(range(pop_size+arch_size)):
            seed = population[i]
            mut = mutations[i]
            evaluate(game,seed,mut,env,avg_rewards,i)

        #save here so we can reconstruct the next generation if need be
        print("saving...")
        np.savez("{}/GAseeds_{}_nmp_{}.npz".format(game,game,gen+1),[population,mutations,avg_rewards],allow_pickle = True)
        print("selecting next gen...")
        select_and_mutate(population,mutations,avg_rewards,pop_size,trunc)




