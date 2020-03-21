from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import os, numpy as np
import cv2

def getModelConv():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model 

def train():
    model = getModelConv()
    (train_im, train_lab), (test_im, test_lab) = mnist.load_data()
    train_im = train_im.reshape((60000, 28, 28, 1))
    train_im = train_im.astype('float32')/255

    test_im = test_im.reshape((10000, 28, 28, 1))
    test_im = test_im.astype('float32')/255

    train_lab = to_categorical(train_lab)
    test_lab = to_categorical(test_lab)


    model.fit(train_im, train_lab, epochs=5, batch_size=60)
    _, test_ac = model.evaluate(test_im, test_lab)

    print ("Finished MNIST Training")
    print ("Accuracy is " + str(test_ac))

    files = os.listdir("test")
    output = to_categorical([int(f.split('.')[0][-1:]) for f in files])

    inp = []
    for f in files:
        src = "test/" + f
        cell = cv2.imread(src)
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(cell)
        #th = np.sum(eh)/(eh.size*4)
        ret, img = cv2.threshold(eq, 23, 255, cv2.THRESH_BINARY_INV)
        img = cv2.resize(img, (28, 28))

        img = img[..., np.newaxis]
        inp.append(img)
    inp = np.array(inp)


    model.fit(inp, output, epochs=2, batch_size=27)

    json = model.to_json()
    with open("modelconv2.json", "w") as f:
        f.write(json)

    model.save_weights("modelconv2.h5")

if __name__ == "__main__":
    train()