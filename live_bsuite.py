import numpy as np
import torch
import matplotlib.pyplot as plt
import functools

def live(agent, environment, num_episodes, max_timesteps, env_name,  
    verbose=False, print_every=10):
    """
    Logic for operating over episodes. 
    """
    observation_data = []
    action_data = []
    rewards = []
    loss = [] 
    std_a0 = []
    std_a1 = []
    std_a2 = []  

    if verbose:
        print("agent: %s, number of episodes: %d" % (str(agent), num_episodes))

    for episode in range(num_episodes):
        agent.reset_cumulative_reward()
        agent.reset_var_tracking_forActions()

        # observation_history is a list of tuples (observation, termination signal)
        timestep = environment.reset()
        reward_history = [] 
        observation_history = [(timestep.observation.flatten(), False)] 
        action_history = []
    
        t = 0
        done = False

        while not timestep.last():
            action = agent.act(observation_history, action_history)
            timestep = environment.step(action)
            reward_history.append(timestep.reward)     

            if timestep.last(): 
                done = True
                observation = timestep.observation.flatten()
            else:
                observation = timestep.observation.flatten()
        
            action_history.append(action)
            observation_history.append((observation, done))
            t += 1
            done = done or (t == max_timesteps)

        agent.update_buffer(observation_history, action_history, reward_history)

        agent.learn_from_buffer() 

        observation_data.append(observation_history)
        action_data.append(action_history)
        rewards.append(agent.cummulative_reward)
        loss.append(agent.running_loss) 

        std_a0.append(agent.std_per_ep_0) 
        std_a1.append(agent.std_per_ep_1) 
        std_a2.append(agent.std_per_ep_2)
        
        if agent.decay_noise_var:
            # linearly decay noise perturbation (sigma in paper) during training
            agent.update_r_var(episode + 1, num_episodes) 

        if verbose and (episode % print_every == 0):
            print("ep %d,  reward %.5f" % (episode, agent.cummulative_reward))

    return observation_data, action_data, rewards, loss, (std_a0, std_a1, std_a2)