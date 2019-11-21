import numpy as np

class FeatureExtractor(object):
    """
    Base feature extractor.
    """
    def __init__(self, **kwargs):
        pass

    def get_feature(self, **kwargs):
        pass

class CartpoleIdentityFeature(FeatureExtractor):
    """
    Return the current state vector.
    Return feature directly as the bsuite observation (8-dim) 
    """
    def __init__(self):
        self.dimension = 8

    def get_feature(self, observation_history):
        return observation_history[-1][0]


class CartpoleNeuralFeature(FeatureExtractor):
    def __init__(self):
        self.dimension = 6 

    def get_feature(self, observation_history):
        # assume obs = (x, x_dot, theta, theta_dot)
        x, x_dot, theta, theta_dot = observation_history[-1][0]  
        features = [np.cos(theta), np.sin(theta), x, x_dot/10.0, theta_dot/10.0, 1. if abs(x) < 1 else 0.]  
        return np.array(features)  


class DeepSeaIdentityFeature(FeatureExtractor):
    """
    Returns the current state vector
    """
    def __init__(self, num_steps):
        self.dimension = 2
        self.feature_space = tuple([(m,n) for m in range(num_steps) for n in range(num_steps)])

    def get_feature(self, observation_history):
        return observation_history[-1][0]

class DeepseaOneHotFeature(FeatureExtractor):
    def __init__(self, num_steps):
        self.num_steps = num_steps  
        self.dimension = num_steps ** 2  

    def get_feature(self, observation_history):
        horizontal, vertical = observation_history[-1][0]
        if vertical == self.num_steps:
            return None 
        else:
            ind = self.num_steps * horizontal + vertical  
            one_hot = np.zeros(self.dimension)  
            one_hot[ind] = 1.0 
            return one_hot   
