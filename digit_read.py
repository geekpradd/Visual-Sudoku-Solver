import cv2
import numpy as np 
from neuralnet import Network

img = cv2.imread("ttt.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dim = (28, 28)
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

l = []
for a in range(28):
    for b in range(28):
        if resized[a,b] <= 127:
            l.append([1])
        else:
            l.append([0])

net = Network([784, 30, 10])

print(np.argmax(net.forward(l)))