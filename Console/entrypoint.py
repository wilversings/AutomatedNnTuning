from keras.models import Sequential
import logging
import json
from KerasWrapper.Wrappers.EvaluationData import EvaluationData
from KerasWrapper.Wrappers.LayerWrapper import LayerWrapper
from KerasWrapper.Wrappers.ArtificialNn import ArtificialNn
from keras.layers.core import Dense
import numpy as np
from KerasWrapper.Tuners.Population import Population

from tensorflow.python.client import device_lib

print("-- Devices list ---------------------\n")
device_lib.list_local_devices()
print('\n')

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    filename='general.log',
                    filemode='w')

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
train_out = np.array(list(map(label_to_out_layer, train_out)))

test_in, test_out = parse_test()
test_out = np.array(list(map(label_to_out_layer, test_out)))

train_in = np.array(train_in) / 255
test_in = np.array(test_in) / 255


pop = Population.from_json(open("populationConfig.json").read())

eval_data = EvaluationData(test_in, test_out, train_in, train_out)

pop.grow_by_nr_of_generations(10, eval_data)
