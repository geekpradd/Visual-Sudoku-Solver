import numpy as np
from mlxtend.data import loadlocal_mnist

neurons = []
biases = []
weights = []
l_size = [784, 100, 10]
num_layers = len(l_size)

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def feedforward(neurons, weights, biases):
	for l in range(1, num_layers):
		neurons[l] = sigmoid(np.dot(weights[l], neurons[l-1].T) + biases[l])
	return neurons

for i in l_size:
	neurons.append(np.full(i, 0.0))

weights = np.load("weights.npz", allow_pickle=True)["arr_0"]
biases = np.load("biases.npz", allow_pickle=True)["arr_0"]

test, lab2 = loadlocal_mnist(
        images_path='t10k-images-idx3-ubyte', 
        labels_path='t10k-labels-idx1-ubyte')

crct = 0
for i in range(10000):
	neurons[0] = test[i]
	feedforward(neurons, weights, biases)

	i_M = np.argmax(neurons[num_layers-1])
	if i_M == lab2[i]:
		crct += 1

print("ACCURACY : " + str(crct) + "/10000")
