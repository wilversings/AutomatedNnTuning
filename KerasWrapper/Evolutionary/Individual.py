from abc import abstractmethod, ABCMeta
import uuid


class Individual(metaclass=ABCMeta):
    
    def __init__(self):
        self._age = 0
        self._name = str(uuid.uuid1())

    @abstractmethod
    def crossover(self, other):
        pass

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def measure_fitness(self, eval_data):
        pass

    @property
    def age(self) -> int:
        return self._age

    @property
    def name(self) -> str:
        return self._name

    def increase_age(self):
        self._age += 1