
import numpy as np
import copy
from torch import full
from torch.distributions.dirichlet import Dirichlet
import random

class GaussianNoise(object):
    def __init__(self, dimension, num_epochs, mu=0.0, var=1):
        self.mu = mu
        self.var = var
        self.dimension = dimension
        self.epochs = 0
        self.num_epochs = num_epochs
        self.min_epsilon = 0.01 # minimum exploration probability
        self.epsilon = 0.3
        self.decay_rate = self.epsilon/num_epochs # exponential decay rate for exploration prob
        self.iter = 0
        self.dist = Dirichlet(full((dimension,),1/dimension))

    def sample(self):
        x = self.dist.sample() * self.epsilon
        self.epsilon = max(self.min_epsilon,self.epsilon-self.decay_rate)
        return x

    def reset(self):
        self.epsilon = 0.3

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mu, self.var)