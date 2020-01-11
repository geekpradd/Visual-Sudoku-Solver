import numpy as np
import cv2
from mlxtend.data import loadlocal_mnist

train, lab = loadlocal_mnist(
        images_path='train-images-idx3-ubyte', 
        labels_path='train-labels-idx1-ubyte')

train = np.float32(train)
lab = np.float32(lab)
lab = np.transpose(lab)

test, lab2 = loadlocal_mnist(
        images_path='t10k-images-idx3-ubyte', 
        labels_path='t10k-labels-idx1-ubyte')

test = np.float32(test)
lab2 = np.float32(lab2)
lab2 = np.transpose(lab2)

#lab2 = lab2[:,np.newaxis]

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, lab)
ret,result,neighbours,dist = knn.findNearest(test,k=5)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==lab2
correct = np.count_nonzero(matches)
accuracy = correct*100.0/matches.size
print(accuracy)