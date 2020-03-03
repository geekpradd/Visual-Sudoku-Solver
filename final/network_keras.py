import numpy as np 
from keras.models import Sequential
from keras.layers import Dense

from loader import load_data_wrapper

train, valid, test = load_data_wrapper()

model = Sequential()

model.add(Dense(784, input_dim=784, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print ("finished compiling")
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

model.fit(inp, out, epochs=15, batch_size=10)

_, accuracy = model.evaluate(test_inp, test_out)
print ("Accuracy is " + str(accuracy))

json = model.to_json()
with open("model.json", "w") as f:
    f.write(json)

model.save_weights("model.h5")