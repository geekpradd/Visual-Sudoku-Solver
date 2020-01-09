from neuralnet import *
from loader import *

net = Network([784, 40, 10])
train, valid, dest = load_data_wrapper()
net.gradient_descent(train, 10, 0.1, 100, 500)