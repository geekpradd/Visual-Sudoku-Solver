import cv2
import numpy as np
from loader import *
from neuralnet_crossentropy import Network
train, valid, dest = load_data_wrapper()
custom_train = []

net = Network([784, 30, 10])

# # for digit in range(10):
# #     ret, img = cv2.threshold(cv2.equalizeHist(cv2.imread("digits/" + str(digit) + ".jpg", 0)), 30, 255, cv2.THRESH_BINARY)
# #     resized = cv2.resize(img, (28, 28))
# #     resized = np.array([np.reshape(x, (1)) for x in np.subtract(255, resized[resized > -1])])

# #     out = np.array([np.array([0]) for x in range(10)])
# #     out[digit] = np.array([1])
# #     custom_train.append((resized, out))

# for digit in range(1, 10):
#     ret, img = cv2.threshold(cv2.equalizeHist(cv2.imread("digits/p" + str(digit) + ".jpg", 0)), 30, 255, cv2.THRESH_BINARY)
#     resized = cv2.resize(img, (28, 28))
#     resized = np.array([np.reshape(x, (1)) for x in np.subtract(255, resized[resized > -1])])

#     out = np.array([np.array([0]) for x in range(10)])
#     out[digit] = np.array([1])
#     custom_train.append((resized, out))


# net.gradient_descent(train, 30, 10, 3.0, 0)

net.test(dest, 1, 5000)