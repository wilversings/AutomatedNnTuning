from KerasWrapper.Evolutionary.Individual import Individual
from KerasWrapper.Wrappers.LayerWrapper import LayerWrapper
from KerasWrapper.Utility.JsonConfigManager import JsonConfigManager
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

    def generation_report(self, i):

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
    def population(self):
        return self._population

    @staticmethod
    def generate_rand_population(pop_size, input_size, output_size, clasfProb, layer_nr_range, layer_size_range, batchSize, epochs):
        return Population([
            copy(ArtificialNn(input_size, output_size, clasfProb))
                .with_batch_size(batchSize)
                .with_epochs(epochs)
                .with_layers([
                    LayerWrapper(
                        size=           randint(*layer_size_range),
                        activation=     "relu",
                        init_weights=   None,
                        init_biases=    None)

                    for _ in range(randint(*layer_nr_range))
                ])
            for _ in range(pop_size)
        ])

    @staticmethod
    def from_blueprint(ann_blueprint: ArtificialNn, ann_list):
        population = [copy(ann_blueprint)
                      .with_batch_size(ann["batchSize"])
                      .with_epochs(ann["epochs"])
                      .with_layers([
                          LayerWrapper(layer["size"], layer["activation"], None, None) for layer in ann["layers"]
                      ])
                      .compile() for ann in ann_list]
        return Population(population)

    @staticmethod
    def from_json(json_config: str):
        config = json.loads(json_config)
        JsonConfigManager.validate_population_config(config)

        if config["type"] == "ArtificialNn":
            return Population.from_blueprint(
                ArtificialNn(
                    config["inputSize"],
                    config["outputSize"],
                    config["clasfProb"]
                ), 
                config["individuals"]
            )

