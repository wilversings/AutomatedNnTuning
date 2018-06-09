from KerasWrapper.Evolutionary.Individual import Individual
from KerasWrapper.Wrappers.LayerWrapper import LayerWrapper
from KerasWrapper.Wrappers.EvaluationData import EvaluationData
from KerasWrapper.Wrappers.ArtificialNn import ArtificialNn
from KerasWrapper.Wrappers.EvaluationData import EvaluationData
from copy import copy
import logging
from KerasWrapper.Utility.Utils import Utils
from KerasWrapper.Evolutionary.EvaluatedIndividual import EvaluatedIndividual
from sortedcontainers import SortedList
from random import random, randint
import json
from math import ceil
from typing import List
import numpy as np

class Population:
    
    PRESSURE_RATE  = 0.3
    ELITISM_RATE = 0.1
    TOURNAMENT_SIZE = 9

    def __init__(self, initial_populaiton: list):
        self._population = None
        self._population_raw = initial_populaiton


        self._graveyard = []
        self._logger = logging.getLogger("population")

    def reproduce(self, eval_data: EvaluationData):
        pop_list = list(reversed(self._population))
        pop_len = len(pop_list)

        elite = ceil(pop_len * Population.ELITISM_RATE)

        mating_pool = \
            [(1, individual) for individual in pop_list[: elite]] + \
            [(1 - i * Population.PRESSURE_RATE / pop_len, individual) for i, individual in list(enumerate(pop_list))[elite: ]]

        _, selected_ind = list(zip(*sorted(mating_pool, reverse=True)[:Population.TOURNAMENT_SIZE]))
       
        sel_len = len(selected_ind)

        new_generation = [
            EvaluatedIndividual(
                selected_ind[i].individual
                .crossover(selected_ind[j].individual)
                .mutate()
                .compile(), 
                eval_data
            )
            for i in range(sel_len - 1)
            for j in range(i + 1, sel_len)
        ]

        self._population = SortedList(new_generation + list(selected_ind))

        self._logger.info("Reproduction: new individuals: %d, total individuals: %d", len(new_generation), len(self._population))

    def generation_report(self, i) -> bool:

        pop_size = len(self._population)

        if pop_size == 0:
            self._logger.info("Your species didn't survive !")
            return False

        self._logger.info("Growing done for generation %d! individuals: %d, best's fitness: %f, avg fitness: %f", 
                            i, pop_size, self._population[-1].fitness, sum(x.fitness for x in self._population) / pop_size)

        return True

    def grow_by_nr_of_generations(self, nr_of_generaitons: int, eval_data: EvaluationData):
        
        self._logger.info("Started growing generation 0...");
        self._population = SortedList(list(map(lambda x: EvaluatedIndividual(x, eval_data), self._population_raw))*3)

        self.generation_report(0)

        for i in range(nr_of_generaitons):

            self._logger.info("Started growing generation %d...", i + 1)
            self.reproduce(eval_data)

            if not self.generation_report(i + 1):
                break

    @property
    def population(self) -> SortedList:
        return self._population

    @staticmethod
    def _set_weights_and_biases(layers: List[LayerWrapper], pop_input_size: int) -> List[LayerWrapper]:
        
        layers[0] = LayerWrapper(
                size=           layers[0].size,
                activation=     'relu',
                init_weights=   np.random.rand(pop_input_size, layers[0].size),
                init_biases=    np.random.rand(layers[0].size)
            )
        for i in range(1, len(layers)):
            layers[i] = LayerWrapper(
                size=           layers[i].size,
                activation=     'relu',
                init_weights=   np.random.rand(layers[i - 1].size, layers[i].size),
                init_biases=    np.random.rand(layers[i].size)
            )
        return layers

    @staticmethod
    def generate_rand_population(
        pop_size:           int, 
        input_size:         int, 
        output_size:        int, 
        clasf_prob:         bool, 
        layer_nr_range:     (int, int), 
        layer_size_range:   (int, int), 
        batch_size:         int, 
        epochs:             int) -> 'Population':
        return Population([
            copy(ArtificialNn(input_size, output_size, clasf_prob))
                .with_batch_size(batch_size)
                .with_epochs(epochs)
                .with_layers(Population._set_weights_and_biases([
                    LayerWrapper(
                        size=           randint(*layer_size_range),
                        activation=     "relu",
                        init_weights=   None,
                        init_biases=    None)

                    for _ in range(randint(*layer_nr_range))
                ], input_size))
                .compile()
            for _ in range(pop_size)
        ])
