import numpy as np
import cv2
# from mlxtend.data import loadlocal_mnist
import random

from gen_train import get_stretched

neurons = []
deltas = []
biases = []
weights = []
l_size = [784, 100, 10]
num_layers = len(l_size)
eta = 0.5
epochs = 30
lambd = 0

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

train = []
lab = []

for i in range(1, 10):
	ret, img = cv2.threshold(cv2.equalizeHist(cv2.imread('digits/p'+str(i)+'.jpg', 0)), 20, 255, cv2.THRESH_BINARY)
	resized = cv2.resize(img, (28, 28))
	ar = np.subtract(255, resized)
	im = cv2.imread('digits/q'+str(i)+'.jpg')
	# print(ar)
	i1 = cv2.resize(get_stretched(im, 1, 2), (28, 28), cv2.INTER_LINEAR)
	i2 = cv2.resize(get_stretched(im, 1, 3), (28, 28), cv2.INTER_LINEAR)
	i3 = cv2.resize(get_stretched(im, 1, 4), (28, 28), cv2.INTER_LINEAR)
	i4 = cv2.resize(get_stretched(im, 2, 3), (28, 28), cv2.INTER_LINEAR)
	# cv2.imshow("image", i3)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	ar = ar[ar > -1]
	i1 = i1[i1 > -1]
	i2 = i2[i2 > -1]
	i3 = i3[i3 > -1]
	i4 = i4[i4 > -1]
	
	train.append(ar)
	train.append(i1)
	train.append(i2)
	train.append(i3)
	train.append(i4)

	lab.append(i)
	lab.append(i)
	lab.append(i)
	lab.append(i)
	lab.append(i)

print (len(train))
# for i in range(0, 10):
# 	ret, img = cv2.threshold(cv2.equalizeHist(cv2.imread('digits/'+str(i)+'.jpg', 0)), 23, 255, cv2.THRESH_BINARY)
# 	resized = cv2.resize(img, (28, 28))
# 	ar = np.subtract(255, resized[resized > -1])
# 	train.append(ar)
# 	lab.append(i)

mini_batch = 9
n_of_mb = 1200
index = []
for i in range(45):
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
	for i in range(45):
		neurons[0] = train[i]
		feedforward(neurons, weights, biases)

		i_M = np.argmax(neurons[num_layers-1])
		if i_M == lab[i]:
			crct += 1
	print("\nEpoch " + str(p+1) + " : " + str(crct) + "/19")

	np.savez("weights", weights)
	np.savez("biases", biases)
