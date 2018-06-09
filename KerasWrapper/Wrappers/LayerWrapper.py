from KerasWrapper.Evolutionary.Individual import Individual
from KerasWrapper.Utility.Utils import Utils
from random import choice, random

class LayerWrapper(Individual):

    MUTATION_CHANCE = 0.2

    def __init__(self, size: int, activation: str, init_weights, init_biases):
        
        # Hyperparameters that are configured by the Evolutive algorithm
        self._size = size
        self._activation = activation
        self._init_weights = init_weights
        self._init_biases = init_biases

    
    def crossover(self, other: 'LayerWrapper') -> 'LayerWrapper':
        return LayerWrapper(size=           (self._size + other._size) // 2,
                            activation=     self._activation if Utils.flip_coin() else other._activation,
                            init_weights=   None,
                            init_biases=    None)

    def mutate(self) -> 'LayerWrapper':
        if random() < self.MUTATION_CHANCE:
            self._size = self._size + choice([-1, 1])
        return self

    def measure_fitness(self):
        raise NotImplementedError("Operation not supported. Cannot measure fitness of an layer")
        
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