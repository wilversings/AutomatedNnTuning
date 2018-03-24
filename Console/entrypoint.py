from keras.models import Sequential
from KerasWrapper.Wrappers.LayerWrapper import LayerWrapper
from KerasWrapper.Wrappers.ArtificialNn import ArtificialNn
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

fitness = ArtificialNn(784, 10, True)\
    .with_batch_size(150)\
    .with_epochs(10)\
    .with_layers([
        LayerWrapper(300, 'relu'),
        LayerWrapper(300, 'relu')
    ])\
    .compile()\
    .measure_fitness(
        np.array(train_in) / 255,
        np.array(train_out),
        np.array(test_in) / 255,
        np.array(test_out)
    )

print(fitness)
