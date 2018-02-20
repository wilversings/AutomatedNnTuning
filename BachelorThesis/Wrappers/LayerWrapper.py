

class LayerWrapper:

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

    def crossover(self):
        pass

    def mutate(self):
        pass