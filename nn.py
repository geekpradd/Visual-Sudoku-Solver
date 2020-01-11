import numpy as np
from mlxtend.data import loadlocal_mnist
import random

neurons = []
deltas = []
biases = []
weights = []
l_size = [784, 100, 10]
num_layers = len(l_size)
eta = 0.5
epochs = 10
lambd = 5.0/60000

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def backprop(neurons, weights, y, deltas, l):
	if l == 0:
		return deltas
	if l == num_layers-1:
		deltas[l] = neurons[l] - y		# cross entropy cost function
	else :
		deltas[l] = np.dot(weights[l+1].T, deltas[l+1]) * neurons[l] * (1-neurons[l])
	return deltas

def calcGrad(neurons, deltas):
	bgrad = []
	for l in range(num_layers):
		bgrad.append(deltas[l])

	wgrad = []
	wgrad.append(np.full((1, 1), 0))
	for l in range(1, num_layers):
		wgrad.append(np.dot(deltas[l][:,None],neurons[l-1][None,:]))
		#wgrad.append(np.dot(np.reshape(neurons[l-1], (-1, 1))), np.reshape(deltas[l], (-1, 1)).T)
	return wgrad, bgrad

def feedforward(neurons, weights, biases):
	for l in range(1, num_layers):
		neurons[l] = sigmoid(np.dot(weights[l], neurons[l-1].T) + biases[l])
	return neurons

# weights.append(np.full((1, 1), 0.0))
for i in l_size:
	neurons.append(np.full(i, 0.0))
	deltas.append(np.full(i, 0.0))
# 	biases.append(np.random.randn(i))
# for i in range(1, len(l_size)):
# 	weights.append(np.random.randn(l_size[i], l_size[i-1]))

weights = np.load("weights.npz", allow_pickle=True)["arr_0"]
biases = np.load("biases.npz", allow_pickle=True)["arr_0"]

train, lab = loadlocal_mnist(
        images_path='train-images-idx3-ubyte', 
        labels_path='train-labels-idx1-ubyte')

test, lab2 = loadlocal_mnist(
        images_path='t10k-images-idx3-ubyte', 
        labels_path='t10k-labels-idx1-ubyte')

mini_batch = 10
n_of_mb = int(60000/mini_batch)
index = []
for i in range(60000):
	index.append(i)

for p in range(epochs):
	random.shuffle(index)
	for i in range(n_of_mb):
		wgradSum = []
		bgradSum = []
		wgradSum.append(np.full((1, 1), 0))
		for l in range(0, num_layers):
			bgradSum.append(np.full(l_size[l], 0.0))
			if l > 0:
				wgradSum.append(np.full((l_size[l], l_size[l-1]), 0.0))

		for j in range(mini_batch):
			neurons[0] = train[index[i*mini_batch + j]]/255.0
			neurons = feedforward(neurons, weights, biases)

			y = np.full(10, 0)
			y[lab[index[i*mini_batch + j]]] = 1
			for level in range(num_layers) :
				deltas = backprop(neurons, weights, y, deltas, num_layers-1-level)

			wgrad, bgrad = calcGrad(neurons, deltas)
			wgradSum = np.add(wgrad, wgradSum)
			bgradSum = np.add(bgrad, bgradSum)

		wgradSum = np.add(np.multiply(lambd, weights), wgradSum)
		for l in range(1, len(l_size)):
			weights[l] = np.subtract(weights[l], wgradSum[l] * eta / mini_batch)
			biases[l] = np.subtract(biases[l], bgradSum[l] * eta / mini_batch)
		
	crct = 0
	for i in range(10000):
		neurons[0] = test[i]
		feedforward(neurons, weights, biases)

		i_M = np.argmax(neurons[num_layers-1])
		if i_M == lab2[i]:
			crct += 1
	print("\nEpoch " + str(p+1) + " : " + str(crct) + "/10000")

	np.savez("weights", weights)
	np.savez("biases", biases)


