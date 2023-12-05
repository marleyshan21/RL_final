import torch.nn as nn
import torch
import math
import numpy as np
import random
import pprint

def variance_initializer_(tensor, scale=1.0, mode='fan_in', distribution='uniform'):
    
    """
    Helps to initialize the weights of the network.
    Variance scaling helps improve the stability of the network 
    by setting the initial weights in an appropriate manner
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling
    
    """
    
    
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        scale /= max(1., fan_in)
    elif mode == "fan_out":
        scale /= max(1., fan_out)
    else:
        raise ValueError

    if distribution == 'uniform':
        limit = math.sqrt(3.0 * scale)
        nn.init.uniform_(tensor, -limit, limit)
    else:
        raise ValueError
    


def set_global_seed(seed):

    """
    Helps to set the seed for the random number generators
    in the torch, numpy and random modules.
    
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_env_seed(env, seed):

    """
    Helps to set the seed for the random number generator
    in the gym environment.
    """
    print("env, seed", env, seed)
    env.seed(seed)
    env.action_space.seed(seed) 
    


def from_nested_dict(data):
    if not isinstance(data, dict):
        return data
    else:
        return AttrDict({key: from_nested_dict(data[key])
                            for key in data})
    

class AttrDict(dict):

    """
    Helps to access the dictionary keys as attributes.
    Useful when we have a nested dictionary.

    Example:
    >>> a = {'a': {'b': 1}}
    >>> a = AttrDict(a)
    >>> a.a.b

    instead of
    >>> a['a']['b']
    """


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for key in self.keys():
            self[key] = from_nested_dict(self[key])

    def __str__(self):
        return pprint.pformat(self.__dict__)