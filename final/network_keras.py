import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

from loader import load_data_wrapper

train, valid, test = load_data_wrapper()

def getModelDense():
    model = Sequential()

    model.add(Dense(784, input_dim=784, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model 

def getModelConv():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())

def train_network():
    inp = []
    out = []
    count = 0
    for i, o in train:
        if len(inp) > 10000:
            break
        base_i = []
        base_o = []
        for item in i:
            base_i.append(item[0])
        
        for item in o:
            base_o.append(item[0])
    
        inp.append(np.asarray(base_i))
        out.append(np.asarray(base_o))

    test_inp = []
    test_out = []
    count = 0
    for i, o in test:
        if len(test_inp) > 1000:
            break
        base_i = []
        base_o = []
        # print ("count is " + str(count))
        # print (o)
        for item in i:
            base_i.append(item[0])
        for digit in range(10):
            if (digit == o):
                base_o.append(1)
            else:
                base_o.append(0)
            
        if (count == 0):
            print (base_o)
        count += 1
        test_inp.append(np.asarray(base_i))
        test_out.append(np.asarray(base_o))

    inp = np.asarray(inp)
    out = np.asarray(out)

    test_inp = np.asarray(test_inp)
    test_out = np.asarray(test_out)
    model = getModel()
    model.fit(inp, out, epochs=15, batch_size=10)

    _, accuracy = model.evaluate(test_inp, test_out)
    print ("Accuracy on MNIST Test Data " + str(accuracy))


    json = model.to_json()
    with open("model.json", "w") as f:
        f.write(json)

    model.save_weights("model.h5")

if __name__ == "__main__":
    train_network()