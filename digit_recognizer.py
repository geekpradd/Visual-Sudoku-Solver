import cv2
import numpy as np

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

for i in range(1, 10):
	ret, img = cv2.threshold(cv2.equalizeHist(cv2.imread('digits/p'+str(i)+'.jpg', 0)), 23, 255, cv2.THRESH_BINARY_INV)
	resized = cv2.resize(img, (28, 28))

	# cv2.imshow("image", resized)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	neurons[0] = np.divide(resized[resized > -1], 255.0)
	neurons = feedforward(neurons, weights, biases)

	print(np.argmax(neurons[num_layers-1]))
	print(neurons[2])
	print("\n")
