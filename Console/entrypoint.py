from keras.models import Sequential
import logging
import json
from KerasWrapper.Wrappers.EvaluationData import EvaluationData
from KerasWrapper.Wrappers.LayerWrapper import LayerWrapper
from KerasWrapper.Wrappers.ArtificialNn import ArtificialNn
from keras.layers.core import Dense
import numpy as np
from KerasWrapper.Tuners.Population import Population
from KerasWrapper.Problems.CharRecognition import CharRecognition

from tensorflow.python.client import device_lib

print("-- Devices list ---------------------\n")
device_lib.list_local_devices()
print('\n')

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    filename='general.log',
                    filemode='w')

problem = CharRecognition('train/mnist', 'train/mnist_labels', 'test/mnist_test', 'test/mnist_test_labels')
test_in, test_out, train_in, train_out = problem.perform_k_fold(10000)

pop = Population.from_json(open("populationConfig.json").read())

eval_data = EvaluationData(test_in, test_out, train_in, train_out)

pop.grow_by_nr_of_generations(10, eval_data)
