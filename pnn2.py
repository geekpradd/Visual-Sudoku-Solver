import numpy as np
import cv2
import random

neurons = []
deltas = []
biases = []
weights = []
l_size = [2500, 100, 100, 10]
num_layers = len(l_size)
eta = 1.0
epochs = 30
lambd = 0.0/13

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

#weights.append(np.full((1, 1), 0.0))
for i in l_size:
	neurons.append(np.full(i, 0.0))
	deltas.append(np.full(i, 0.0))
# 	biases.append(np.random.randn(i))
# for i in range(1, len(l_size)):
# 	weights.append(np.random.randn(l_size[i], l_size[i-1]))

weights = np.load("weights2.npz", allow_pickle=True)["arr_0"]
biases = np.load("biases2.npz", allow_pickle=True)["arr_0"]

train = []
lab = []

for i in range(1, 10):
	ret, img = cv2.threshold(cv2.equalizeHist(cv2.imread('digits/p'+str(i)+'.jpg', 0)), 23, 255, cv2.THRESH_BINARY)
	img = cv2.resize(img, (50, 50))
	ar = np.subtract(255, img)
	ret, img2 = cv2.threshold(cv2.equalizeHist(cv2.imread('digits/'+str(i)+'.jpg', 0)), 23, 255, cv2.THRESH_BINARY)
	img2 = cv2.resize(img2, (50, 50))
	ar2 = np.subtract(255, img2)
	ret, img3 = cv2.threshold(cv2.equalizeHist(cv2.imread('digits/q'+str(i)+'.jpg', 0)), 23, 255, cv2.THRESH_BINARY)
	img3 = cv2.resize(img2, (50, 50))
	ar3 = np.subtract(255, img3)
	train.append(ar[ar > -1])
	train.append(ar2[ar2 > -1])
	train.append(ar3[ar3 > -1])
	lab.append(i)
	lab.append(i)
	lab.append(i)

# for i in range(0, 10):
# 	ret, img = cv2.threshold(cv2.equalizeHist(cv2.imread('digits/'+str(i)+'.jpg', 0)), 23, 255, cv2.THRESH_BINARY)
# 	resized = cv2.resize(img, (28, 28))
# 	ar = np.subtract(255, resized[resized > -1])
# 	train.append(ar)
# 	lab.append(i)

mini_batch = 13
n_of_mb = 400
index = []
for i in range(27):
	index.append(i)

for p in range(epochs):
	for i in range(n_of_mb):
		random.shuffle(index)
		wgradSum = []
		bgradSum = []
		wgradSum.append(np.full((1, 1), 0))
		for l in range(0, num_layers):
			bgradSum.append(np.full(l_size[l], 0.0))
			if l > 0:
				wgradSum.append(np.full((l_size[l], l_size[l-1]), 0.0))

		for j in range(mini_batch):
			neurons[0] = train[index[j]]/255.0
			neurons = feedforward(neurons, weights, biases)

			y = np.full(10, 0)
			y[lab[index[j]]] = 1
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
	for i in range(27):
		neurons[0] = train[i]
		feedforward(neurons, weights, biases)

		i_M = np.argmax(neurons[num_layers-1])
		if i_M == lab[i]:
			crct += 1
	print("\nEpoch " + str(p+1) + " : " + str(crct) + "/27")

	np.savez("weights2", weights)
	np.savez("biases2", biases)
