import numpy as np
from keras.models import Sequential, model_from_json
from neuralnet_crossentropy import Network
from loader import load_data_wrapper

import cv2 

KERAS = False 
def load_model():
    f = open("model.json", "r")
    json = f.read()
    f.close()
    model = model_from_json(json)

    model.load_weights("model.h5")


    train, valid, test = load_data_wrapper()

    test_inp = []
    test_out = []
    count = 0
    for i, o in test:
        if len(test_inp) > 2000:
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
            
        count += 1
        test_inp.append(np.asarray(base_i))
        test_out.append(np.asarray(base_o))

    test_inp = np.asarray(test_inp)
    test_out = np.asarray(test_out)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def recognize(img):
    inp = np.array([np.divide(img[img > -1], 255.0)])
    ar = []
    for elem in inp[0]:
        ar.append([elem])
    ar = np.array(ar)

    
    if KERAS:
        model = load_model()
        res = model.predict(inp)[0]
    else:
        net = Network([784, 30, 10])
        res = net.forward(ar)
    
    return res

# _, accuracy = model.evaluate(test_inp, test_out)
# print ("Accuracy is " + str(accuracy))
