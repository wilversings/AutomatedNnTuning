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

    @abstractmethod
    def crossover(self, counterpart: NeuralNetWrapper):
        
        self_layers = self._layers
        ctp_layers = counterpart._layers

        if len(self_layers) > len(ctp_layers):
            # Make sure that self_layers has less layers than ctp_layers
            self_layers, ctp_layers = ctp_layers, self_layers

        # Chose random samples from the net with more layers
        ctp_layers = random.sample(ctp_layers, len(self_layers))

        return NeuralNetWrapper(self._input_size, self._output_size)\
            .with_layers(
                # Crossover the choices
                list(map(lambda x, y: x.crossover(y).mutate(), ctp_layers, self_layers))
            )\
            .with_batch_size(
                # Chose one random batch size from the two parts
                # TODO: try with average
                random.choice([self._batch_size, counterpart._batch_size])
            )\
            .with_epochs(
                random.choice([self._epochs, counterpart._epochs])
            )
    
    @abstractmethod
    def mutate(self):
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
    



