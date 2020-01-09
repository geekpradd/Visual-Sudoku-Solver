from neuralnet import *
from loader import *

net = Network([784, 30, 10])
train, valid, dest = load_data_wrapper()

# net.gradient_descent(train, 30, 10, 3.0)
net.test(dest, 1, 5000)
