from multiprocessing.pool import Pool

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

        self.pool = Pool(processes=6)

    @staticmethod
    def evaluate_individual(individual_and_data):
        return EvaluatedIndividual(individual_and_data[0], individual_and_data[1])

    def reproduce(self, eval_data: EvaluationData):
        pop_list = list(reversed(self._population))
        pop_len = len(pop_list)

        elite = ceil(pop_len * Population.ELITISM_RATE)

        mating_pool = \
            [(1, individual) for individual in pop_list[: elite]] + \
            [(1 - i * Population.PRESSURE_RATE / pop_len, individual) for i, individual in list(enumerate(pop_list))[elite: ]]

        _, selected_ind = list(zip(*sorted(mating_pool, reverse=True)[:Population.TOURNAMENT_SIZE]))
       
        sel_len = len(selected_ind)

        new_generation = list(self.pool.map(self.evaluate_individual, ((selected_ind[i].individual
                .crossover(selected_ind[j].individual)
                .mutate(), eval_data) for i in range(sel_len - 1)
            for j in range(i + 1, sel_len))))

        for ind in new_generation:
            logging.getLogger("fitness").info("{} was born! fitness: {}".format("Name: someone", ind.fitness))
        self._population = SortedList(new_generation + list(selected_ind))

        self._logger.info("Reproduction: new individuals: %d, total individuals: %d", len(new_generation), len(self._population))

    def generation_report(self, i, nr_of_generations) -> bool:

        pop_size = len(self._population)

        if pop_size == 0:
            self._logger.info("Your species didn't survive !")
            return False

        print("Growing generation {}/{} done!".format(i, nr_of_generations))
        self._logger.info("Growing done for generation %d! individuals: %d, best's fitness: (%f, %f), avg fitness: (%f, %f)",
                            i, pop_size, self._population[-1].fitness[0], self._population[-1].fitness[1], sum(x.fitness[0] for x in self._population) / pop_size, sum(x.fitness[1] for x in self._population) / pop_size)

        return True

    def grow_by_nr_of_generations(self, nr_of_generaitons: int, eval_data: EvaluationData):
        
        self._logger.info("Started growing generation 0...")

        self._population = SortedList(
            list(
                self.pool.map(
                    self.evaluate_individual,
                    ((x, eval_data) for x in self._population_raw)
                )
            )
        )
        for ind in self._population:
            logging.getLogger("fitness").info("{} was born! fitness: {}".format("Name: someone", ind.fitness))

        self.generation_report(0, nr_of_generaitons)

        for i in range(nr_of_generaitons):

            self._logger.info("Started growing generation %d...", i + 1)
            self.reproduce(eval_data)

            if not self.generation_report(i + 1, nr_of_generaitons):
                break

    @property
    def population(self) -> SortedList:
        return self._population

    @staticmethod
    def _create_layers(sizes: List[int], pop_input_size: int) -> List[LayerWrapper]:
        
        layers = [LayerWrapper(
                size=           sizes[0],
                activation=     'relu',
                init_weights=   np.random.rand(pop_input_size, sizes[0]),
                init_biases=    np.random.rand(sizes[0])
            )]
        for i in range(1, len(sizes)):
            layers.append(LayerWrapper(
                size=           sizes[i],
                activation=     'relu',
                init_weights=   np.random.rand(sizes[i - 1], sizes[i]),
                init_biases=    np.random.rand(sizes[i])
            ))
        return layers

    @staticmethod
    def generate_rand_population(
        pop_size:           int,
        input_size:         int,
        output_size:        int,
        layer_nr_range:     (int, int),
        layer_size_range:   (int, int),
        batch_size:         int,
        epochs:             int) -> 'Population':
        return Population([
            copy(ArtificialNn(input_size, output_size))
                .with_batch_size(batch_size)
                .with_epochs(epochs)
                .with_layers(Population._create_layers([
                    randint(*layer_size_range)
                    for _ in range(randint(*layer_nr_range))
                ], input_size))
            for _ in range(pop_size)
        ])
