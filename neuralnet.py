# Implements a digit recognising neural network
import numpy as np 
import random

class Network:
    def __init__(self, params):
        # params is a list containing sizes layer wises
        self.layers = len(params)
        self.biases = [np.random.randn(siz, 1) for siz in params[1:]] # first layer won't have bias 
        #to do check if the param should have a 1 (bias should be a row vector)
        self.weights = [np.random.randn(siz, prev) for siz, prev in zip(params[1:], params[:-1])]
    
    def gradient_descent(self, training_data, cycles, eta, batch_size, num_batches):
        # group data into batches of batch_size
        # training data has elements that have two numpy arrays: input layer values and output layer values
        # num batches refers to the number of mini batches that will be used in stochastic gradient descent
        # to get better averaging we do this grouping cycles number of times
        n = len(training_data)
        for iter in range(cycles):
            mini_batches = [training_data[s:s+batch_size] for s in range(0, n, batch_size)]

            count = 0
            for batch in mini_batches:
                base_w = [np.zeros(w.shape) for w in self.weights]
                       # random.shuffle(training_data)    
                base_b = [np.zeros(b.shape) for b in self.biases]
                for dataset in batch:
                    
                    # do back propagation for this dataset
                    # average out this to obtain the gradient   
                    change_w, change_b = self.back_prop(dataset)
                    base_w =  [w + ch for w, ch in zip(base_w, change_w)] 
                    base_b =  [b + ch for b, ch in zip(base_b, change_b)]
                   
                # we have the average gradient 
                self.weights = [w-(eta*ch/len(batch)) for w, ch in zip(self.weights, base_w)]
                self.biases = [b-(eta*ch/len(batch)) for b, ch in zip(self.biases, base_b)]
                count += 1
                print ("Finished batch {0}".format(count))

    def test(self, training_data, l, r):
        i = l
        success = 0
        total = 0
        while i<=r:
            result = self.forward(training_data[i][0])
            best_val = 0
            best = -1
            j = 0
            actual = -1
            while j<=9:
                if result[j] > best_val:
                    best_val = result[i]
                    best = j
                if training_data[i][1][j] > 0.5:
                    actual = j
                j+=1

            for term, actual in zip(result, training_data[i][1]):
                net_cost += (term-actual)*(term-actual)
            net_cost /= len(result)
            
            if actual == best:
                success+=1
            total += 1
        
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

    def back_prop(self, dataset):
        activations = [dataset[0]]
        zs = []
        a = dataset[0]
        for weight, bias in zip(self.weights, self.biases):
            # print(bias.shape)
            zs.append(np.dot(weight, a) + bias)
            a = self.sigmoid(np.dot(weight, a) + bias)
            activations.append(a)

        layers = len(self.weights) + 1
        delta = 2*(activations[-1]-dataset[1])*self.sigmoid_prime(zs[-1])
        change_bias = self.biases 
        change_weight = self.weights 

        change_bias[layers-2] = delta 
        # currently operating on weights at layers-2-iter
        for j in range(len(change_weight[layers-2])):
            for k in range(len(change_weight[layers-2][j])):
                change_weight[layers-2][j][k] = activations[layers-2][k]*delta[j]

        # want to return gradients layer wise 
        for iter in range(layers-3):
            delta = np.dot(self.weights[layers-2-iter].transpose(), delta)*self.sigmoid_prime(zs[layers-3-iter])        
            change_bias[layers-3-iter] = delta 
            # currently operating on weights at layers-2-iter
            for j in range(len(change_weight[layers-3-iter])):
                for k in range(len(change_weight[layers-3-iter][j])):
                    change_weight[layers-3-iter][j][k] = activations[layers-3-iter][k]*delta[j]
            # back propagate delta       

        
        return change_weight, change_bias
        