from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical

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

    print ("Accuracy is " + str(test_ac))

    json = model.to_json()
    with open("modelconv.json", "w") as f:
        f.write(json)

    model.save_weights("modelconv.h5")

if __name__ == "__main__":
    train()