# Implements a digit recognising neural network
import numpy as np 
from math import e 
import random

class Network:
    def __init__(self, params):
        # params is a list containing sizes layer wises
        self.layers = len(params)
        self.biases = [np.random.randn(siz) for siz in params[1:]] # first layer won't have bias 
        #to do check if the param should have a 1 (bias should be a row vector)
        self.weights = [np.random.randn(siz, prev) for siz, prev in zip(params[1:], parans[:-1])]
    
    def gradient_descent(self, training_data, cycles, eta, batch_size, num_batches):
        # group data into batches of batch_size
        # training data has elements that have two numpy arrays: input layer values and output layer values
        # num batches refers to the number of mini batches that will be used in stochastic gradient descent
        # to get better averaging we do this grouping cycles number of times
        n = len(training_data)
        for iter in range(cycles):
            random.shuffle(training_data) 
            mini_batches = [training_data[s:s+batch_size] for s in range(0, n, batch_size)]

            for batch in mini_batches:
                for dataset in batch:
                    # do back propagation for this dataset
                    # average out this to obtain the gradient   
                    change_w, change_b = self.back_prop(dataset)

    def sigmoid(self, vector):
        #returns sigmoid of a vector
        return 1.0/1.0 + np.exp(-vector)

    def sigmoid_prime(self, vector):
        return self.sigmoid(vector)*(1-self.sigmoid(vector))

    def forward(self, a):
        # if a is the input layer, returns the resultant at the final end 
        for weight, bias in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(weight, a) + bias)   
        return a

    def back_prop(self, dataset):
        activations = [dataset[0]]
        zs = []
        a = dataset[0]
        for weight, bias in zip(self.weights, self.biases):
            zs.append(np.dot(weight, a) + bias)
            a = self.sigmoid(np.dot(weight, a) + bias)
            activations.append(a)

        delta = [2*(a-y)*self.sigmoid_prime()]
        




