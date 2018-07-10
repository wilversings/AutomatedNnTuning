from KerasWrapper.Evolutionary.Individual import Individual
from KerasWrapper.Wrappers.EvaluationData import EvaluationData
import logging

class EvaluatedIndividual:

    def __init__(self, individual: Individual, eval_data: EvaluationData):
        self._fitness = individual.measure_fitness(eval_data)
        self._individual = individual

    def __gt__(self, ctp):
        return self._fitness > ctp._fitness

    def __lt__(self, ctp):
        return self._fitness < ctp._fitness

    @property
    def fitness(self):
        return self._fitness

    @property
    def individual(self):
        return self._individual



