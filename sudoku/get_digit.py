import numpy as np
from keras.models import Sequential, model_from_json
import cv2, os
dir_path = os.path.dirname(os.path.realpath(__file__))

KERAS = True 
CONV = True 

def load_conv():
    f = open(dir_path + "/models/modelconv2.json", "r")
    json = f.read()
    f.close()
    model = model_from_json(json)

    model.load_weights(dir_path + "/models/modelconv2.h5")
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

def load_model():
    f = open(dir_path + "/models/model.json", "r")
    json = f.read()
    f.close()
    model = model_from_json(json)

    model.load_weights(dir_path + "/models/model.h5")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def recognize(img):
    inp = np.array([np.divide(img[img > -1], 255.0)])
    ar = []
    for elem in inp[0]:
        ar.append([elem])
    ar = np.array(ar)

    trans = img[..., np.newaxis]
    
    if CONV:
        model = load_conv()
        res = model.predict(np.array([trans]))[0]
    elif KERAS:
        model = load_model()
        res = model.predict(inp)[0]
    
    return res

# _, accuracy = model.evaluate(test_inp, test_out)
# print ("Accuracy is " + str(accuracy))
