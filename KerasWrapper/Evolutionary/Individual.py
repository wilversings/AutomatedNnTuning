from abc import abstractmethod, ABCMeta


class Individual(metaclass=ABCMeta):
    
    @abstractmethod
    def crossover(self, other):
        pass

    @abstractmethod
    def mutate(self):
        pass
