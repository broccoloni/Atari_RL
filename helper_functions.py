import torch
import numpy as np
from AtariNet import *
import gym
from tqdm import tqdm
from PIL import Image

#To process the observations from gym
def ProcessIm(obs,prev_obs):
    obs1 = obs
    obs2 = prev_obs
    observation = torch.Tensor(obs1 - obs2)
    observation.unsqueeze_(0)
    observation.unsqueeze_(0)
    return observation

#To watch the agents play when they're trained
def make_gameplay_gif(params,agent):
    game = params[0]
    maxt = params[4]
    numactions = params[7]
    
    env = gym.make(f'ALE/{game}',render_mode = 'rgb_array')
    env = gym.wrappers.GrayScaleObservation(env)
    env.metadata['render_fps'] = 25
                      
    obs, info = env.reset()
    prev_obs = np.zeros(obs.shape)

    gameplay = []

    for t in range(maxt):
        #Preprocessing
        observation = ProcessIm(obs,prev_obs)    

        #Forward pass
        probs = agent.forward(observation)

        #Sample action space from probability distribution
        action = np.random.choice(numactions, 1, p = probs.detach().numpy())[0]

        #Take action
        prev_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)

        img = env.render().astype(np.uint8)
        im = Image.fromarray(img)
        gameplay.append(im)

        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t+1))
            break
            
    if not (terminated or truncated):
        print(f"Episode did not finish, agent was still playing after {maxt} timesteps")
        
    #convert sequence of images to gif
    gameplay[0].save('gameplay.gif',save_all=True, append_images=gameplay[1:], optimize=False, duration=40, loop=0)

#Genetic algorithm functions

#Evaluate a given mutation
def evaluate_GA(params,mut,env):
    #create initial model
    numactions = params[2]
    inshape = params[3]
    mut_power = params[4]
    test_size = params[7]
    poolsizes = params[9]
    numconvlayers = params[10]
    maxt = params[11]
    seed = params[12]

    torch.manual_seed(seed)
    atarinet = AtariNetCONV(inshape=inshape,
                            poolsizes = poolsizes,
                            numconvlayers = numconvlayers, 
                            outsize = numactions)

    #apply mutations
    atarinet.mutate(mut_power,mut)
            
    total_rewards = []
    for i_episode in range(test_size):
        obs, info = env.reset()
        prev_obs = np.zeros(obs.shape)
        rewards = []
        for t in range(maxt):
#             env.render(mode='human')

            #Preprocessing
            observation = ProcessIm(obs,prev_obs)    

            #Forward pass
            probs = atarinet.forward(observation)

            #Sample action space from probability distribution
            action = np.random.choice(numactions, 1, p = probs.detach().numpy())[0]

            #Take action
            prev_obs = obs
            obs, reward, terminated, truncated, info = env.step(action)
                
            #Store values of episode
            rewards.append(reward)
            
            #Game over
            if terminated or truncated:
                break
            
        total_rewards.append(np.sum(rewards))
    avg_rewards = np.mean(total_rewards)
    return avg_rewards

#Selection and Mutation algorithm
def select_and_mutate(params,population,mutations,avg_pop_rewards):
    pop_size = params[5]
    arch_size = params[6]
    numsurvivors = params[8]
    seed = params[12]
    
    fit_order = np.argsort(avg_pop_rewards)[::-1] #best to worst

    print("Average generation rewards:\t",np.mean(avg_pop_rewards))
    print("Min reward:\t\t",avg_pop_rewards[fit_order[-1]])
    print("Max reward:\t\t",avg_pop_rewards[fit_order[0]])
    print() 

    #Probabilistic selection based off of rewards received
    #selection_probs = np.exp(avg_rewards[fit_order])/np.sum(np.exp(avg_rewards))
    #survivors = np.random.choice(pop_size+arch_size,pop_size,p = selection_probs)

    #cutoff selection choosing top numsurvivors of the population
    np.random.seed(seed = seed)
    survivors = fit_order[np.random.randint(0,numsurvivors,size = pop_size)]
    survivors = np.append(survivors,fit_order[:arch_size])
    population = [population[i] for i in survivors]    

    mutations = [deepcopy(mutations[i]) for i in survivors]
    for i in range(pop_size):
        mutations[i].append(np.random.randint(seed))
        
    return population,mutations

#Load a generation of the genetic algorithm
def load_gen(game,cur_gen):
    data = np.load("{}/GAseeds_{}_nmp_{}.npz".format(game,game,cur_gen),allow_pickle = True)
    data = [data[key] for key in data]
    data = data[1]
    population = data[0]
    mutations = data[1]
    avg_pop_rewards = np.array(data[2],dtype = float)
    return population,mutations,avg_pop_rewards

def train_GA(num_gens,params,population,mutations,env):    
    game = params[0]
    cur_gen = params[1]
    pop_size = params[5]
    arch_size = params[6]
    
    print("Starting GA:")
    
    for gen in range(cur_gen,cur_gen+num_gens):
        print("Generation:\t{}/{} (generation {})".format(gen+1-cur_gen,num_gens,gen+1))
        avg_pop_rewards = []
        for i in tqdm(range(pop_size+arch_size)):
            seed = population[i]
            mut = mutations[i]
            avg_rewards = evaluate_GA(params,mut,env)
            avg_pop_rewards.append(avg_rewards)
            
        #save here so we can reconstruct the next generation if need be
        np.savez("{}/GAseeds_{}_nmp_{}.npz".format(game,game,gen+1),[population,mutations,avg_pop_rewards],allow_pickle = True)
        population,mutations = select_and_mutate(params,population,mutations,avg_pop_rewards)


#Policy Gradient Functions

def evaluate_PG(params,env,actor,critic):
    maxt = params[3]
    numactions = params[7]
    
    ep_rewards = 0  #in case of sparse rewards, like tetris
    while ep_rewards == 0:
        logprobs = []
        rewards = []
        rewards_est = []
        obs, info = env.reset()
        prev_obs = np.zeros(obs.shape)

        for t in range(maxt):
#             env.render(mode='human')

            #Preprocessing
            observation = ProcessIm(obs,prev_obs)    

            #Forward pass
            probs = actor.forward(observation)
            reward_est = critic.forward(observation)

            #Sample action space from probability distribution
            action = np.random.choice(numactions, 1, p = probs.detach().numpy())[0]

            #Take action
            prev_obs = obs
            obs, reward, terminated, truncated, info = env.step(action)

            #break the sparse rewards loop
            if reward != 0:
                ep_rewards = 1

            #Store values of episode
            logprobs.append(torch.log(probs[action]))
            rewards.append(reward)
            rewards_est.append(reward_est)

            if terminated or truncated:
                batch_ep_lens.append(t)
                break

        return logprobs,rewards,rewards_est,t

def backprop(params,batch_metrics):
    #Backprop - One step actor critic method from Sutton & Barto book
    #just do batch size 1 right now, implement batches later
    batch_size = params[3]
    gamma = params[5]
    
    batch_logprobs = batch_metrics[0]
    batch_rewards = batch_metrics[1]
    batch_rewards_est = batch_metrics[2]
    batch_ep_lens = batch_metrics[3]
             
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

    sum_r = torch.sum(batch_rewards).detach()
    sum_r_est = torch.sum(batch_rewards_est).detach()
    blens = batch_len
    return sum_r,sum_r_est,blens
    
def save_PG(i_batch,params,measures,actor,actorOptimizer,critic,criticOptimizer):
    game = params[0]
    midsize = params[6]
    actor.save("PGsaves/{}_m{}_{}.tar".format(game,midsize,i_batch),actorOptimizer,measures = measures)
    critic.save("PGsaves/Critic_{}_m{}_{}.tar".format(game,midsize,i_batch),criticOptimizer, measures = measures)
    
def train_PG(params,sum_rewards,sum_rewards_est,batch_lens,env,actor, actorOptimizer,critic,criticOptimizer):
    cur_batches = params[1]
    num_batches = params[2]
    batch_size = params[3]
    maxt = params[4]
    
    for i_batch in tqdm(range(cur_batches,cur_batches+num_batches)):
        batch_logprobs = []
        batch_rewards = []
        batch_rewards_est = []
        batch_ep_lens = []

        for i_episode in range(batch_size):
            logprobs,rewards,rewards_est,t = evaluate_PG(params,env,actor,critic)
            batch_logprobs.append(logprobs)
            batch_rewards.append(rewards)
            batch_rewards_est.append(rewards_est)
            batch_ep_lens.append(t)

        batch_metrics = [batch_logprobs,
                         batch_rewards,
                         batch_rewards_est,
                         batch_ep_lens]

        sum_r,sum_r_est,blens = backprop(params,batch_metrics)
        sum_rewards.append(sum_r)
        sum_rewards_est.append(sum_r_est)
        batch_lens.append(blens)
        
        if i_batch != cur_batches and i_batch%100 == 0:
            measures = [sum_rewards,sum_rewards_est,batch_lens]
            save_PG(i_batch,params,measures,actor,actorOptimizer,critic,criticOptimizer)
