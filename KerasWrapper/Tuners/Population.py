from KerasWrapper.Evolutionary.Individual import Individual
from KerasWrapper.Evolutionary.EvaluatedIndividual import EvaluatedIndividual
from sortedcontainers import SortedList

class Population:
    

    def __init__(self, initial_populaiton: list[Individual]):

        self._population = SortedList(map(lambda x: EvaluatedIndividual(x)))

    def reproduce(self):
        pass

    def natural_selection(self):
        pass

    def grow(number_of_generaitons):
        pass

