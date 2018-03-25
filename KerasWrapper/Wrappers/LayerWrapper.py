from KerasWrapper.Evolutionary.Individual import Individual
from KerasWrapper.Utils import Utils
from random import choice, random

class LayerWrapper(Individual):

    MUTATION_CHANCE = 0.2

    def __init__(self, size: int, activation):
        
        # Hyperparameters that are configured by the Evolutive algorithm
        self._size = size
        self._activation = activation

    @property
    def size(self):
        return self._size

    @property
    def activation(self):
        return self._activation
    
    def crossover(self, other):
        return LayerWrapper((self._size + other._size) // 2,
                            self._activation if Utils.flip_coin() else other._activation)

    def mutate(self):
        if random() < self.MUTATION_CHANCE:
            self._size = self._size + choice([-1, 1])
        return self

    def measure_fitness(self):
        raise NotImplementedError("Operation not supported. Cannot measure fitness of an layer")
