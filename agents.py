'''
Agents for cartpole
'''
import numpy as np
import torch
import torch.nn as nn
import math

class Buffer(object):
    """
    A finite-memory buffer that rewrites oldest data when buffer is full.
    Stores tuples of the form (feature, action, reward, next feature). 
    """
    def __init__(self, size=50000):
        self.size = size
        self.buffer = []
        self.next_idx = 0

    def add(self, x, a, r, x_next):
        if self.next_idx == len(self.buffer):
            self.buffer.append((x, a, r, x_next))
        else:
            self.buffer[self.next_idx] = (x, a, r, x_next)
        self.next_idx = (self.next_idx + 1) % self.size

    def sample(self, batch_size=1):
        idxs = np.random.randint(len(self.buffer), size=batch_size)
        return [self.buffer[i] for i in idxs]


class RandomizedBuffer(Buffer):
    def __init__(self, size=50000):

        Buffer.__init__(self, size)  

    def add(self, x, a, r, x_next, b):
        if self.next_idx == len(self.buffer):
            self.buffer.append((x, a, r, x_next, b)) 
        else:
            self.buffer[self.next_idx] = (x, a, r, x_next, b) 
        self.next_idx = (self.next_idx + 1) % self.size   


class TensorBuffer(object):
    """
    A finite-memory buffer that rewrites oldest data when buffer is full.
    Stores tuples of the form (feature, action, reward, next feature). 
    """
    def __init__(self, feature_dim, num_heads, size=50000):
        self.size = size
        self.next_idx = 0
        self.valid_size = 0
        self.x = torch.zeros(size, feature_dim, dtype=torch.float)
        self.a = torch.zeros(size, dtype=torch.long)
        self.r = torch.zeros(size, dtype=torch.float)
        self.x_next = torch.zeros(size, feature_dim, dtype=torch.float)
        self.not_terminal = torch.zeros(size, dtype=torch.float)
        self.num_heads = num_heads  

        if num_heads > 1:
            self.mask = torch.zeros(size, num_heads, dtype=torch.float) 
        else:
            self.mask = None     

    def add(self, x, a, r, x_next):
        self.x[self.next_idx].copy_(torch.tensor(x, dtype=torch.float))
        self.a[self.next_idx].copy_(torch.tensor(a, dtype=torch.long))
        self.r[self.next_idx].copy_(torch.tensor(r, dtype=torch.float))
        if x_next is not None:
            self.x_next[self.next_idx].copy_(torch.tensor(x_next, dtype=torch.float))
            self.not_terminal[self.next_idx].copy_(torch.tensor(1, dtype=torch.float))
        if self.mask is not None:
            # update the mask at next_idx: each data transition is included with prob 0.5 
            self.mask[self.next_idx].copy_(torch.tensor(np.random.binomial(1, 0.5, self.num_heads), dtype=torch.float))  

        self.next_idx = (self.next_idx + 1) % self.size
        self.valid_size += 1

    def sample(self, batch_size=1):
        idxs = np.random.randint(min(self.valid_size, self.size), size=batch_size)
        if self.mask is None: 
            return (
                self.x[idxs],
                self.a[idxs],
                self.r[idxs],
                self.x_next[idxs],
                self.not_terminal[idxs],
            )
        else:
            return (
                self.x[idxs],
                self.a[idxs],
                self.r[idxs],
                self.x_next[idxs],
                self.not_terminal[idxs],
                self.mask[idxs], 
            )

    def to(self, device):
        self.x = self.x.to(device)
        self.a = self.a.to(device)
        self.r = self.r.to(device)
        self.x_next = self.x_next.to(device)
        self.not_terminal = self.not_terminal.to(device)
    
        if self.mask is not None:
            self.mask = self.mask.to(device)

class Agent(object):
    """
    generic class for agent
    """
    def __init__(self, action_set, reward_function):
        self.action_set = action_set
        self.reward_function = reward_function
        self.cummulative_reward = 0

    def __str__(self):
        pass

    def reset_cumulative_reward(self):
        self.cummulative_reward = 0

    def update_buffer(self, observation_history, action_history):
        pass

    def learn_from_buffer(self):
        pass

    def act(self, observation_history, action_history):
        pass

    def get_episode_reward(self, observation_history, action_history):
        tau = len(action_history)
        reward_history = np.zeros(tau)
        for t in range(tau):
            reward_history[t] = self.reward_function(
                observation_history[:t+2], action_history[:t+1])
        return reward_history

    def _random_argmax(self, action_values):
        argmax_list = np.where(action_values==np.max(action_values))[0]
        return self.action_set[argmax_list[np.random.randint(argmax_list.size)]]

    def _epsilon_greedy_action(self, action_values, epsilon):
        if np.random.random() < 1- epsilon:
            return self._random_argmax(action_values)
        else:
            return np.random.choice(self.action_set, 1)[0]

    def _boltzmann_action(self, action_values, beta):
        action_values = action_values - max(action_values)
        action_probabilities = np.exp(action_values / beta)
        action_probabilities /= np.sum(action_probabilities)
        return np.random.choice(self.action_set, 1, p=action_probabilities)[0]

    def _epsilon_boltzmann_action(self, action_values, epsilon):
        action_values = action_values - max(action_values)
        action_probabilities = np.exp(action_values / (np.exp(1)*epsilon))
        action_probabilities /= np.sum(action_probabilities)
        return np.random.choice(self.action_set, 1, p=action_probabilities)[0]

class RandomAgent(Agent):
    """
    selects actions uniformly at random from the action set
    """
    def __str__(self):
        return "random agent"

    def act(self, observation_history, action_history):
        return np.random.choice(self.action_set, 1)[0]

    def update_buffer(self, observation_history, action_history):
        reward_history = self.get_episode_reward(observation_history, action_history)
        self.cummulative_reward += np.sum(reward_history)


class MLP(nn.Module):
    def __init__(self, dimensions, isPrior = False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dimensions)-1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i+1]))

        self.isPrior = isPrior

    def forward(self, x):
        for l in self.layers[:-1]:
            x = nn.functional.relu(l(x))        
        x = self.layers[-1](x)
        return x

    def initialize(self):
        # not used by default 
        for i, m in enumerate(self.layers):
            if i == 0:
                torch.nn.init.normal_(m.weight)
            else:
                torch.nn.init.xavier_uniform_(m.weight)

class MLP_std(nn.Module):
    def __init__(self, dimensions, isPrior = False):
        super(MLP_std, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dimensions)-1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i+1]))

        # can use self-defined initialize() to init weights  
        # if isPrior:
        #     self.initialize()
            
        self.isPrior = isPrior 

    def forward(self, x):
        for l in self.layers[:-1]:            
            x = nn.functional.relu(l(x))

        x = nn.functional.softplus(self.layers[-1](x))
        return x

    def initialize(self):
        for i, m in enumerate(self.layers):
            if i == 0:
                torch.nn.init.normal_(m.weight)
            else:
                torch.nn.init.xavier_uniform_(m.weight)

class IndDQNAgent(Agent):
    def __init__(self, action_set, reward_function, feature_extractor, 
        hidden_dims=[50, 50], hidden_dims_std = [50, 50], learning_rate=5e-4, buffer_size=50000, 
        batch_size=64, num_batches=100, starts_learning=5000, discount=0.99, target_freq=10, 
        verbose=False, print_every=1, test_model_path=None, num_heads = 10, index_std = 1.0, prior_beta = 2.0,
        act_resample_z = False, r_perturb_scale = 1.0, decay_noise_var = False, gpu_id=None, prior_beta_std=None):                        

        Agent.__init__(self, action_set, reward_function)
        self.feature_extractor = feature_extractor
        self.feature_dim = self.feature_extractor.dimension

        self.num_heads = num_heads 
        self.index_sigma = index_std    
        
        self.prior_beta = prior_beta   
        self.prior_beta_std = self.prior_beta if prior_beta_std is None else prior_beta_std  
      
        self.act_resample_z = act_resample_z  # for training with TS (False) or not (True)      
   
        self.hidden_dims = hidden_dims  
        self.hidden_dims_std = hidden_dims_std   

        self.batch_size = batch_size         
        self.use_gpu = gpu_id is not None
        self.device = torch.device('cuda:{}'.format(gpu_id)) if self.use_gpu else torch.device('cpu')  

        self.z_generator = torch.distributions.normal.Normal(loc = torch.tensor([0.0]), scale = torch.tensor([self.index_sigma]))   

        # build Q network
        # we use a multilayer perceptron
        # fit two nets: one for mean, one for std, with W2 loss; Q = mean + std * Z (Gaussian)  
        dims = [self.feature_dim] + hidden_dims + [len(self.action_set)]  
        dims_std = [self.feature_dim] + hidden_dims_std + [len(self.action_set) * self.num_heads]

        self.model_mean = MLP(dims) 
        self.model_std = MLP_std(dims_std)  
        self.prior_mean = MLP(dims, isPrior = True)
        self.prior_std = MLP_std(dims_std, isPrior = True) 
             
        self.prior_mean.eval() 
        self.prior_std.eval()

        if self.use_gpu:
            self.model_mean.to(self.device) 
            self.model_std.to(self.device)
            self.prior_mean.to(self.device)
            self.prior_std.to(self.device)

        if self.num_heads == 1:  
            self.z_act = np.random.normal(0, self.index_sigma, 1)
        else:
            self.z_act = np.random.normal(0, self.index_sigma, 1)
            self.picked_m_ind = np.random.choice(self.num_heads, 1)[0]

        self.r_var_multiplier = r_perturb_scale  # add extra std. to perturb rewards
        self.decay_noise_var = decay_noise_var   # boolean variable, whether to decay r_var_multiplier during training 
        self.r_var_decay_rate = None      

        if test_model_path is None:
            self.test_mode = False
            self.learning_rate = learning_rate
            self.buffer_size = buffer_size
            self.num_batches = num_batches
            self.starts_learning = starts_learning
            self.timestep = 0
            self.discount = discount
            self.buffer = TensorBuffer(feature_dim = self.feature_dim,
                                                 num_heads = self.num_heads,
                                                 size = self.buffer_size)  
            self.target_mean = MLP(dims)  
            self.target_std = MLP_std(dims_std)   

            self.target_mean.load_state_dict(self.model_mean.state_dict()) 
            self.target_mean.eval() 

            self.target_std.load_state_dict(self.model_std.state_dict()) 
            self.target_std.eval()

            if self.use_gpu:
                self.buffer.to(self.device)
                self.target_mean.to(self.device)
                self.target_std.to(self.device) 

            self.optimizer_mean = torch.optim.Adam(self.model_mean.parameters(), lr = self.learning_rate)   
            self.optimizer_std = torch.optim.Adam(self.model_std.parameters(), lr = self.learning_rate) 
 
            self.target_freq = target_freq # target nn updated every target_freq episodes
            self.num_episodes = 0

            # for debugging purposes
            self.verbose = verbose
            self.running_loss = 1.
            self.print_every = print_every

            # for tracking the std. of action [0, 1, 2] (on the online network)
            self.std_per_ep_0 = 0. 
            self.std_per_ep_1 = 0. 
            self.std_per_ep_2 = 0.        

        else:
            self.test_mode = True
            self.load_models(test_model_path)

    def __str__(self):
        return "parameterized_indexed_nets"

    def reset_var_tracking_forActions(self):
        self.std_per_ep_0 = 0. 
        self.std_per_ep_1 = 0. 
        self.std_per_ep_2 = 0.  

    def update_r_var(self, episode_num, num_episodes):
        if self.r_var_decay_rate is None:
            self.r_var_end = 1.0             
            self.r_var_decay_rate = (self.r_var_multiplier - self.r_var_end)/(0.9 * num_episodes)   
        else:
            # do decay based on decay_rate 
            self.r_var_multiplier = max(self.r_var_multiplier - self.r_var_decay_rate, self.r_var_end)             

    def update_z_for_acting(self):
        if self.num_heads == 1:
            self.z_act = np.random.normal(0, self.index_sigma, 1) 
        else:
            self.z_act = np.random.normal(0, self.index_sigma, 1)
            self.picked_m_ind = np.random.choice(self.num_heads, 1)[0] 


    def update_buffer(self, observation_history, action_history, reward_history=None):   
        """
        update buffer with data collected from current episode
        and update the sampled z variable which will be used for the next episode 
        """

        tau = len(action_history)
        
        # update z:
        self.update_z_for_acting()                                                       

        if reward_history is None:
            reward_history = self.get_episode_reward(observation_history, action_history)
            self.cummulative_reward += np.sum(reward_history)
        else:
            reward_history = np.array(reward_history)
            self.cummulative_reward += np.sum(reward_history)     
        
        feature_history = np.zeros((tau+1, self.feature_extractor.dimension))

        for t in range(tau+1):
            feature_history[t] = self.feature_extractor.get_feature(observation_history[:t+1])

        for t in range(tau-1):
            self.buffer.add(feature_history[t], action_history[t], 
                reward_history[t], feature_history[t+1])  
        done = observation_history[tau][1]
        if done:
            feat_next = None
        else:
            feat_next = feature_history[tau]
        self.buffer.add(feature_history[tau-1], action_history[tau-1], 
            reward_history[tau-1], feat_next)       


    def act(self, observation_history, action_history):
        """ select action according to a greedy policy with respect to 
        the sampled Q network """

        feature = self.feature_extractor.get_feature(observation_history)

        with torch.no_grad():
            mu = self.model_mean(torch.from_numpy(feature).float()).numpy()   
            m = self.model_std(torch.from_numpy(feature).float()).numpy() 

            mu0 = self.prior_mean(torch.from_numpy(feature).float()).numpy() 
            m0 = self.prior_std(torch.from_numpy(feature).float()).numpy()   

            if self.num_heads == 1:   
                action_values = mu + m * self.z_act + self.prior_beta * mu0 + self.prior_beta_std * m0 * self.z_act
            else:
                # use self.picked_m_ind 
                action_values = mu + np.reshape(m, [-1, self.num_heads])[:, self.picked_m_ind] * self.z_act + self.prior_beta * mu0 + self.prior_beta_std * (np.reshape(m0, [-1, self.num_heads])[:, self.picked_m_ind]) * self.z_act

        if not self.test_mode:
            action = self._random_argmax(action_values)
            self.timestep += 1

            # resample z every time:
            if self.act_resample_z:
                # if True: resample Z.
                self.update_z_for_acting()   # z and picked_m_ind are both updated
                
        else:
            # test mode:
            action = self._random_argmax(action_values)
            if self.act_resample_z:
                self.update_z_for_acting()

        return action


    def save(self, path=None, agent_path=None):
        if path is None:
            path = './pins.pt'

        torch.save({
            'model_mean': self.model_mean.state_dict(), 
            'model_std': self.model_std.state_dict(),
            'prior_mean': self.prior_mean.state_dict(),
            'prior_std': self.prior_std.state_dict() 
            }, path)


    def load_models(self, path):
        model_path = path + 'pins.pt'
        checkpoint = torch.load(model_path)  

        self.model_mean.load_state_dict(checkpoint['model_mean'])
        self.model_std.load_state_dict(checkpoint['model_std']) 

        self.prior_mean.load_state_dict(checkpoint['prior_mean'])
        self.prior_std.load_state_dict(checkpoint['prior_std']) 

        self.model_mean.eval()
        self.model_std.eval()
        self.prior_mean.eval()
        self.prior_std.eval()


    def learn_from_buffer(self):
        
        if self.timestep < self.starts_learning:
            return    

        for _ in range(self.num_batches):

            sampled_z_values = self.z_generator.sample(sample_shape=[self.batch_size]).repeat_interleave(len(self.action_set), dim = 1) 
         
            if self.num_heads == 1: 
                feature_batch, action_batch, reward_batch, next_feature_batch, not_terminal_batch = \
                    self.buffer.sample(batch_size=self.batch_size)
            else:  
                feature_batch, action_batch, reward_batch, next_feature_batch, not_terminal_batch, mask_batch = \
                    self.buffer.sample(batch_size=self.batch_size)

                # randomly select one head to train, and get the dropout mask for that head
                selected_heads = (torch.rand_like(mask_batch) * 0.5 + mask_batch).max(1)[1] # (batch_size, ) 
                new_mask_batch = torch.zeros_like(mask_batch)
                new_mask_batch[range(mask_batch.size()[0]), selected_heads] = 1. 

                mask_batch = new_mask_batch   
                
            # Q = mu + mz + beta(mu0 + m0 z)  
            mus = self.model_mean(feature_batch)  
            ms = self.model_std(feature_batch) 

            prior_mus = self.prior_mean(feature_batch)

            prior_ms = self.prior_std(feature_batch)  

            mean_estimates = mus + self.prior_beta * prior_mus

            if self.num_heads == 1:
                std_estimates = ms + self.prior_beta_std * prior_ms
            else:
                ms = torch.reshape(ms, [-1, len(self.action_set), self.num_heads]) 
                prior_ms = torch.reshape(prior_ms, [-1, len(self.action_set), self.num_heads])  
                std_estimates = ms + self.prior_beta_std * prior_ms    #(batch_size, act_size, dim of head)   

            with torch.no_grad():
                if self.num_heads == 1: 
                    stds_per_batch = (torch.min(std_estimates, dim=0)[0]).numpy()  
                else:
                    stds_per_batch = torch.mean(std_estimates,dim=-1, keepdim=False) # (batch_size, action_dim)
                    stds_per_batch = (torch.min(stds_per_batch,dim=0)[0]).numpy() # min over batch 

                self.std_per_ep_0 += stds_per_batch[0]
                self.std_per_ep_1 += stds_per_batch[1]
                if len(self.action_set) > 2:
                    self.std_per_ep_2 += stds_per_batch[2]

            # training:  
            Q_estimates = mean_estimates.gather(1, action_batch.unsqueeze(1)) # (self.batch_size, 1) 

            if self.num_heads == 1: 
                Q_std_estimates = std_estimates.gather(1, action_batch.unsqueeze(1))   # for m(s, a) fitting
            else:
                
                Q_std_estimates = std_estimates[torch.arange(std_estimates.size(0)), action_batch]   


            # compute targets  
            next_means = self.target_mean(next_feature_batch) + self.prior_beta * self.prior_mean(next_feature_batch)

            if self.num_heads == 1:
                next_stds = self.target_std(next_feature_batch) + self.prior_beta_std * self.prior_std(next_feature_batch) 

                # sampled_next_Qs = next_means + next_stds * sampled_z_values 
                # tilde_next_as = sampled_next_Qs.max(1)[1] 

                tilde_next_as = next_means.max(1)[1]   

                next_Q_values = next_means.gather(1, tilde_next_as.unsqueeze(1)).detach().squeeze(1) 
                next_Q_stds = next_stds.gather(1, tilde_next_as.unsqueeze(1)).detach().squeeze(1)

                next_Q_targets = reward_batch.float() + self.discount * (next_Q_values * not_terminal_batch)   
                mean_loss = nn.functional.mse_loss(Q_estimates.squeeze(1), next_Q_targets)   

                next_Q_std_targets = self.r_var_multiplier * self.index_sigma + self.discount * (torch.abs(next_Q_stds) * not_terminal_batch)
                std_loss = nn.functional.mse_loss(Q_std_estimates.squeeze(1), next_Q_std_targets)     
                
            else:
                # multiple heads version
                m_next = torch.reshape(self.target_std(next_feature_batch), [-1, len(self.action_set), self.num_heads])  
                prior_m_next = torch.reshape(self.prior_std(next_feature_batch), [-1, len(self.action_set), self.num_heads])
                
                next_std_vecs = m_next + self.prior_beta_std * prior_m_next  #(batch_size, action_size, dim_head)
                next_std_vecs_t = torch.transpose(next_std_vecs, 1, 2) #(batch_size, dim_head, action_size) 
                selected_next_std_est = next_std_vecs_t[torch.arange(next_std_vecs_t.size(0)), selected_heads].detach()  #(batch_size, action_size)
  
                # sampled_next_Qs = next_means + selected_next_std_est * sampled_z_values
                # tilde_next_as = sampled_next_Qs.max(1)[1]    

                # mean argmax
                tilde_next_as = next_means.max(1)[1]  

                next_Q_values = next_means.gather(1, tilde_next_as.unsqueeze(1)).detach().squeeze(1) 
                next_Q_targets = reward_batch.float() + self.discount * (next_Q_values * not_terminal_batch) 
                mean_loss = nn.functional.mse_loss(Q_estimates.squeeze(1), next_Q_targets)   

                # for std part:
                chosen_next_std_vecs = next_std_vecs[torch.arange(next_std_vecs.size(0)), tilde_next_as].detach()  #(batch_size, dim_head)
                target_std_vecs = self.r_var_multiplier * self.index_sigma + self.discount * (torch.abs(chosen_next_std_vecs) * not_terminal_batch.reshape([-1, 1]).repeat([1, self.num_heads])) 
 
                std_loss = torch.mean(torch.sum( ((Q_std_estimates - target_std_vecs) ** 2) * mask_batch, dim= -1 ))     

            self.optimizer_mean.zero_grad() 
            self.optimizer_std.zero_grad()

            mean_loss.backward() 
            std_loss.backward() 

            self.optimizer_mean.step() 
            self.optimizer_std.step()    

            self.running_loss = 0.99 * self.running_loss + 0.01*(mean_loss.item() + std_loss.item())   

        self.num_episodes += 1

        self.std_per_ep_0 /= (self.num_batches * 1.0) 
        self.std_per_ep_1 /= (self.num_batches * 1.0)
        self.std_per_ep_2 /= (self.num_batches * 1.0)

        if self.verbose and (self.num_episodes % self.print_every == 0):  
            print("indexed dqn ep %d, running loss %.2f, std_0 %.2f, std_1 %.2f, std_2 %.2f" % 
                (self.num_episodes, self.running_loss, self.std_per_ep_0, self.std_per_ep_1, self.std_per_ep_2))    
            
        if self.num_episodes % self.target_freq == 0:
            self.target_mean.load_state_dict(self.model_mean.state_dict()) 
            self.target_std.load_state_dict(self.model_std.state_dict())   

            if self.verbose:
                print("indexed dqn ep %d update two target networks" % self.num_episodes)


def cartpole_reward_function(observation_history, action_history, 
    reward_type='height', move_cost=0.05):
    """
    If the reward type is 'height,' agent gets a reward of 1 + cosine of the
    pole angle per step. Agent also gets a bonus reward of 1 if pole is upright
    and still. 
    If the reward type is 'sparse,' agent gets 1 if the pole is upright 
    and still and if the cart is around the center. 
    There is a small cost for applying force to the cart. 
    """
    state, terminated = observation_history[-1]
    x, x_dot, theta, theta_dot = state
    action = action_history[-1]

    reward = - move_cost * np.abs(action - 1.)

    if not terminated:
        up = math.cos(theta) > 0.95
        still = np.abs(theta_dot) <= 1
        centered = (np.abs(x) <= 1) and (np.abs(x_dot) <= 1)

        if reward_type == 'height':
            reward += math.cos(theta) + 1 + (up and still)

        elif reward_type == 'sparse':
            reward += (up and still and centered)

    return reward

# ----------------------------------------------------------------
# reward function for deep sea experiments hard exploration env. 
# ---------------------------------------------------------------
def deep_sea_reward(observation_history,action_history,horizon,treasure = True,move_cost = 0.01):
    state = observation_history[-1][0]
    prev_state = observation_history[-2][0]
    # horizontal, vertical = state

    reward =0
    if state[0]-prev_state[0] ==1:
    # for deepsea we penalize 'right' action
        if prev_state[0]==prev_state[1]:
            reward = -move_cost / (horizon * 1.0)
    if state[1] == horizon:
        if state[0] == horizon:
            if treasure:
                reward += 1
            else:
                reward += -1
    return reward 