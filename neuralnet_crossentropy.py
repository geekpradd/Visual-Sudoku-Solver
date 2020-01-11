from __future__ import division

import numpy as np 
import random

LOAD = False

# add L2 regularization

class Network(object):
    def load(self):
        self.biases = (np.load("bias.dat", allow_pickle=True)).tolist()
        self.weights = (np.load("weights.dat", allow_pickle=True)).tolist()

    def __init__(self, params):
        # params is a list containing sizes layer wises
        self.layers = len(params)
        self.biases = [np.random.randn(siz, 1) for siz in params[1:]] # first layer won't have bias 
        #to do check if the param should have a 1 (bias should be a row vector)
        self.weights = [np.random.randn(siz, prev) for siz, prev in zip(params[1:], params[:-1])]
    
        if LOAD:
            self.load()
    
    def gradient_descent(self, training_data, cycles, batch_size, eta, lmbda):
        # group data into batches of batch_size
        # training data has elements that have two numpy arrays: input layer values and output layer values
        # num batches refers to the number of mini batches that will be used in stochastic gradient descent
        # to get better averaging we do this grouping cycles number of times
        n = len(training_data)
        for iter in range(cycles):
            random.shuffle(training_data)
            mini_batches = [training_data[s:s+batch_size] for s in range(0, n, batch_size)]

            count = 0
            for batch in mini_batches:
                base_w = [np.zeros(w.shape) for w in self.weights]
                       # random.shuffle(training_data)    
                base_b = [np.zeros(b.shape) for b in self.biases]
                for dataset in batch:
                    # do back propagation for this dataset
                    # average out this to obtain the gradient   
                    change_b, change_w = self.back_prop(dataset[0], dataset[1])
                    base_w =  [w + ch  for w, ch in zip(base_w, change_w)] 
                    base_b =  [b + ch for b, ch in zip(base_b, change_b)]
                   
                # we have the average gradient 
                self.weights = [(1-eta*lmbda/len(batch))*w-((eta/len(batch))*ch) for w, ch in zip(self.weights, base_w)]
                self.biases = [b-((eta/len(batch))*ch) for b, ch in zip(self.biases, base_b)]
                count += 1
                # print ("Finished batch {0}".format(count))

            print ("Finished epoch " + str(iter+1))
        
        weight_np = np.array(self.weights)
        bias_np = np.array(self.biases)

        weight_np.dump("weights.dat")
        bias_np.dump("bias.dat")

    def test(self, test_data, l, r):
        i = l
        success = 0
        total = 0
        while i<=r:
            result = self.forward(test_data[i][0])
            best_val = 0
            best = -1
            j = 0
            actual = test_data[i][1]
            while j<=9:
                if result[j] > best_val:
                    best_val = result[j]
                    best = j
                j+=1
            if actual == best:
                success+=1
            total += 1

            i+=1

        print ("Success: {0}/{1}".format(success, total))

    def sigmoid(self, vector):
        #returns sigmoid of a vector
        return 1.0/(1.0 + np.exp(-vector))

    def sigmoid_prime(self, vector):
        return self.sigmoid(vector)*(1-self.sigmoid(vector))

    def forward(self, a):
        # if a is the input layer, returns the resultant at the final end 
        for weight, bias in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(weight, a) + bias)   
        return a

    def back_prop(self, inp, out):
        activations = [inp]
        zs = []
        a = inp
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, a) + bias
            zs.append(z)
            a = self.sigmoid(z)
            activations.append(a)

        layers = self.layers

        delta = activations[-1] - out
        change_bias = [np.zeros(b.shape) for b in self.biases] 
        change_weight = [np.zeros(w.shape) for w in self.weights]    

        change_bias[-1] = delta 
        change_weight[-1] = np.dot(delta, activations[-2].transpose())

        # want to return gradients layer wise 
        for iter in range(2, layers):
            delta = np.dot(self.weights[-iter + 1].transpose(), delta)*self.sigmoid_prime(zs[-iter])        
            change_bias[-iter] = delta 
            change_weight[-iter] = np.dot(delta, activations[-iter-1].transpose())
                 
        return (change_bias, change_weight)
    