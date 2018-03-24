from LayerWrapper import LayerWrapper
import random
from typing import List
from abc import abstractmethod
from abc import ABC
from keras.models import Sequential


class NeuralNetWrapper(ABC):

    def __init__(self, input_size, output_size):
        
        # Hyperparameters that are configured by the Evolutive algorithm
        self._epochs = None
        self._batch_size = None
        self._layers = None

        # Hyperparameters that are configured by the Evolutive algorithm's user
        self._input_size = input_size
        self._output_size = output_size

    @abstractmethod
    def compile(self):
        pass

    def with_epochs(self, epochs):
        self._epochs = epochs
        return self

    def with_batch_size(self, batch_size):
        self._batch_size = batch_size
        return self

    def with_layers(self, layers: List[LayerWrapper]):

        if __debug__:
            assert(len(layers) >= 2)

        self._layers = layers
        return self
    



