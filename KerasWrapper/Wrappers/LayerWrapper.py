from KerasWrapper.Evolutionary.Individual import Individual
from KerasWrapper.Utility.Utils import Utils
from random import choice, random
from typing import List
import numpy as np
from functools import reduce

class LayerWrapper(Individual):

    MUTATION_CHANCE = 0.2

    def __init__(self, size: int, activation: str, init_weights, init_biases):
        
        # Hyperparameters that are configured by the Evolutive algorithm
        self._size = size
        self._activation = activation
        self._init_weights = init_weights
        self._init_biases = init_biases

    def crossover(self, others: List['LayerWrapper']) -> 'LayerWrapper':
        ans_size = np.mean([self._size] + [x._size for x in others], dtype=int)
        others_weights_mean = reduce(np.matmul, (x._init_weights for x in others)) / len(others)

        # self._init_weights * other_weights_mean
        self_rows, self_cols = self._init_weights.shape
        other_rows, other_cols = others_weights_mean.shape

        weights = (np.matmul(self._init_weights, np.ones((self_cols, other_cols))) +\
        np.matmul(np.ones((self_rows, other_rows)), others_weights_mean)) / 2

        assert(weights.shape[0] == self_rows and weights.shape[1] == other_cols)

        return LayerWrapper(size=           ans_size,
                            activation=     choice([self._activation] + [x._activation for x in others]),
                            init_weights=   weights,
                            init_biases=    np.ones(ans_size))

    def mutate(self) -> 'LayerWrapper':
        # if random() < self.MUTATION_CHANCE:
        #     self._size = self._size + choice([-1, 1])
        return self

    def measure_fitness(self):
        raise NotImplementedError("Operation not supported. Cannot measure fitness of a layer")
        
    @property
    def size(self) -> int:
        return self._size

    @property
    def activation(self) -> str:
        return self._activation

    @property
    def init_weights(self):
        return self._init_weights

    @property
    def init_biases(self):
        return self._init_biases

    def __repr__(self):
        return str(self._size)