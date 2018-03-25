from abc import abstractmethod, ABCMeta


class Individual(metaclass=ABCMeta):
    
    def __init__(self):
        self._age = 0

    @abstractmethod
    def crossover(self, other):
        pass

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def measure_fitness(self):
        pass

    @property
    def age(self):
        return self._age

    def increase_age(self):
        self._age += 1