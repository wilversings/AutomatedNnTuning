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
from random import random
import json

class Population:
    
    AGE_STRETCH = 3

    def __init__(self, initial_populaiton: list):
        self._population = None
        self._population_raw = initial_populaiton

        self._graveyard = []
        self._logger = logging.getLogger("population")

    #@staticmethod
    #def are_selected_for_reproduction(first, second, all):
    #    if __debug__:
    #        assert(first < second)
    #        assert(second < all)

    #    original_chance = (first + second) / 2 * all;
    #    delta_chance = 1 / (second - first) * original_chance

    #    return Utils.uneven(delta_chance)

    def are_selected_for_reproduction(self, i: int, j: int):
        n = len(self._population)
        chance = i * j / (n - 1) ** 2
        return Utils.uneven(chance)

    def is_selected_for_death(self, individual: EvaluatedIndividual, i: int):
        n = len(self._population)
        chance = individual.individual.age / Population.AGE_STRETCH * (1 - i / n)
        return Utils.uneven(chance)

    def reproduce(self, eval_data: EvaluationData):
        pop_list = list(self._population)
        pop_len = len(pop_list)

        new_generation = [
            EvaluatedIndividual(
                pop_list[i].individual
                .crossover(pop_list[j].individual)
                .mutate()
                .compile(), 
                eval_data
            )
            for i in range(pop_len - 1)
            for j in range(i + 1, pop_len)
            if self.are_selected_for_reproduction(i, j)
        ]
            
        for individual in pop_list:
            individual.individual.increase_age()

        self._population.update(new_generation)

        self._logger.info("Reproduction: new individuals: %d, total individuals: %d", len(new_generation), len(self._population))

    def replace(self):
        selected = filter(lambda x: self.is_selected_for_death(x[1], x[0]), enumerate(self._population))
        for sel in selected:
            self._graveyard.append(sel[1])
            self._population.discard(sel[1])

            self._logger.info("{} died!".format(sel[1].individual.name))

    def grow_by_nr_of_generations(self, nr_of_generaitons: int, eval_data: EvaluationData):
        
        self._logger.info("Started growing generation 0...");
        self._population = SortedList(map(lambda x: EvaluatedIndividual(x, eval_data), self._population_raw))
        self._logger.info("Growing done for generation 0! individuals: %d, best's fitness: %f", len(self._population), self._population[-1].fitness)

        for i in range(nr_of_generaitons):

            self._logger.info("Started growing generation %d...", i + 1)
            self.reproduce(eval_data)
            self.replace()

            pop_size = len(self._population)
            self._logger.info("Growing done for generation %d! individuals: %d, best's fitness: %f, avg fitness: %f", 
                              i + 1, pop_size, self._population[-1].fitness, sum(x.fitness for x in self._population) / pop_size)

    @property
    def population(self):
        return self._population

    @staticmethod
    def from_blueprint(ann_blueprint: ArtificialNn, ann_list):
        population = [copy(ann_blueprint)
                      .with_batch_size(ann["batchSize"])
                      .with_epochs(ann["epochs"])
                      .with_layers([
                          LayerWrapper(layer["size"], layer["activation"]) for layer in ann["layers"]
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

