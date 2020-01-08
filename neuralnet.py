import numpy as np 
import random

class Network:
    def __init__(self, params):
        # params is a list containing sizes layer wises
        self.layers = len(params)
        self.biases = [np.random.randn(siz, 1) for siz in params[1:]] # first layer won't have bias 
        self.weights = [np.random.randn(siz, prev) for siz, prev in zip(params[1:], parans[:-1])]
    
    def gradient_descent(self, training_data, cycles, eta, batch_size, num_batches):
        # group data into batches of size num_batches
        # to get better averaging we do this grouping cycles number of times
        for iter in range(cycles):
            random.shuffle(training_data) 
            # training data contains two row vectors
            mini_batches = [training_data[s:s+batch_size] for s in range()]