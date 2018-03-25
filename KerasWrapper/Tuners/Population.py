from KerasWrapper.Evolutionary.Individual import Individual
from KerasWrapper.Utils import Utils
from KerasWrapper.Evolutionary.EvaluatedIndividual import EvaluatedIndividual
from sortedcontainers import SortedList
from random import random

class Population:
    
    AGE_STRETCH = 10

    def __init__(self, initial_populaiton: list[Individual]):

        self._population = SortedList(map(lambda x: EvaluatedIndividual(x)))

    #@staticmethod
    #def are_selected_for_reproduction(first, second, all):
    #    if __debug__:
    #        assert(first < second)
    #        assert(second < all)

    #    original_chance = (first + second) / 2 * all;
    #    delta_chance = 1 / (second - first) * original_chance

    #    return Utils.uneven(delta_chance)

    def are_selected_for_reproduction(self, i, j):
        n = len(self._population)
        chance = i * j / (n - 1) ** 2
        return Utils.uneven(chance)

    def is_selected_for_death(self, individual: EvaluatedIndividual, i):
        n = len(self._population)
        chance = individual.individual.age / Population.AGE_STRETCH * (1 - i / n)
        return Utils.uneven(chance)

    def reproduce(self):
        
        pop_list = list(self._population)
        pop_len = len(pop_list)

        new_generation = [pop_list[i].crossover(pop_list[j]).mutate() 
                for i in range(pop_len - 1) 
                for j in range(i + 1, pop_len) 
                if self.are_selected_for_reproduction(i, j)]
            
        for individual in pop_list:
            individual.increase_age()

        self._population.update(new_generation)

    def replace(self):

        selected = filter(lambda x: self.is_selected_for_death(x[1], x[0]), enumerate(self._population))
        for sel in selected:
            self._population.discard(sel)


    def grow(number_of_generaitons):
        pass

