from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np

from tensorflow.python.client import device_lib

print("-- Devices list ---------------------\n")
device_lib.list_local_devices() 
print('\n')

def parse_train():
    with open('train/mnist', 'rb') as mnist:
        mnist.seek(0x10)
        bytes = np.ndarray.astype(np.array(bytearray(mnist.read())), 'int16')

    with open('train/mnist_labels', 'rb') as labels:
        labels.seek(8)
        label = bytearray(labels.read())

    mnist = np.array_split(bytes, 60000)

    assert(mnist[-1].shape[0] == 28 * 28)
    assert(len(mnist) == 60000 and len(label) == 60000)

    return mnist, label

def parse_test():
    with open('test/mnist_test', 'rb') as mnist:
        mnist.seek(0x10)
        bytes = np.ndarray.astype(np.array(bytearray(mnist.read())), 'int16')

    with open('test/mnist_test_labels', 'rb') as labels:
        labels.seek(8)
        label = bytearray(labels.read())

    mnist = np.array_split(bytes, 10000)

    assert(len(mnist) == 10000 and len(label) == 10000)
    return mnist, label

def label_to_out_layer(label):

    ans = [0] * 10
    ans[label] = 1
    return ans

train_in, train_out = parse_train();
train_out = list(map(label_to_out_layer, train_out))

test_in, test_out = parse_test()
test_out = list(map(label_to_out_layer, test_out))

model = Sequential();
model.add(Dense(units=300, input_dim=784, activation='relu'))
model.add(Dense(units=300, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("-- Training ---------------------\n")
model.fit(np.array(train_in) / 255,
          np.array(train_out),
          epochs=10, batch_size=150, verbose=2)
print("\n")



print("-- Testing ---------------------\n")
loss_and_metrics = model.evaluate(np.array(test_in) / 255,
                                  np.array(test_out),
                                  verbose=2)

print("\n")
print(loss_and_metrics)

print("Done")
