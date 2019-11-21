import numpy as np
import torch
import os
import functools
import matplotlib.pyplot as plt
import bsuite
from bsuite import sweep
import dm_env
from tqdm import trange
from live_bsuite import live
from agents import RandomAgent
from agents import IndDQNAgent 
from agents import cartpole_reward_function  
from feature import CartpoleIdentityFeature, CartpoleNeuralFeature


if __name__ == '__main__':
  
    result_path = './results/cartpole/' 
    agent_path = './agents/cartpole/'

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(agent_path):
        os.makedirs(agent_path)

    # use cartpole_swingup/19 from bsuite, set env. parameters to the ones used in the paper:  
    bsuite_id = 'cartpole_swingup/19'
    sweep.SETTINGS[bsuite_id]['x_reward_threshold'] = 1.0  
    sweep.SETTINGS[bsuite_id]['x_threshold'] = 5.
    sweep.SETTINGS[bsuite_id]['move_cost'] = 0.05         
    
    # train agent over multiple seeds
    for seed in trange(81, 86, 1):  
        env = bsuite.load_and_record(bsuite_id, result_path + str(seed), overwrite = True) 
        np.random.seed(seed)
        torch.manual_seed(seed)

        agent = IndDQNAgent(
            action_set=[0, 1, 2],
            reward_function=functools.partial(cartpole_reward_function, reward_type='sparse'),
            feature_extractor = CartpoleIdentityFeature(), # use feature from bsuite without any modification 
            hidden_dims=[50, 50, 50],
            hidden_dims_std = [50, 50, 50],  
            learning_rate= 1e-3,     
            buffer_size = int(1e6),       
            batch_size = 64,          
            num_batches = 100,  # number of updates per episode         
            starts_learning = 2000,     
            discount = 0.99,
            target_freq = 10,  # frequency to update target nets     
            verbose = True, 
            print_every = 1,   # print result every print_every during training 
            num_heads = 2,     # number of output heads in the uncertainty net, U in paper 
            index_std = 1.0,   # std. for random index Z, default Z ~ N(0, 1)     
            prior_beta= 2.0,      # prior beta_1 in paper
            prior_beta_std = 2.0, # prior beta_2 in paper   
            r_perturb_scale = 2.0,  # initial perturbation scale, sigma in paper     
            act_resample_z = False,  # if True: resample Z within episodes; if False, acting as TS.  
            decay_noise_var = True)  # if True, the sigma added to noise decays from r_perturb_scale * index_std to 1.0 * index_std during learning.    
            
        num_episodes = 4000   
        max_step = 50000 #not in use, max_step is determined by bsuite env. to 1000 for cartpole    
            
        _, _, rewards, loss, stds = live(
            agent=agent,
            environment=env,
            num_episodes=num_episodes,    
            max_timesteps=max_step,  
            verbose=True,
            print_every=1, 
            env_name = bsuite_id.split('/')[0])

        # store data and agent
        # bsuite also generates and stores resuilts in .csv in result_path/[seed]  
        file_name = '|'.join(['pins_l', str(seed)])
        np.save(os.path.join(result_path, file_name), loss)

        file_name = '|'.join(['pins_r', str(seed)])
        np.save(os.path.join(result_path, file_name), rewards) 

        file_name = '|'.join(['pins', str(seed)])
        agent.save(path=os.path.join(agent_path, file_name+'.pt'), agent_path=agent_path)

        