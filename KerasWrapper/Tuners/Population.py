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
    
    AGE_STRETCH = 10

    def __init__(self, initial_populaiton: list):
        self._population = None
        self._population_raw = initial_populaiton

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

    def replace(self):
        selected = filter(lambda x: self.is_selected_for_death(x[1], x[0]), enumerate(self._population))
        for sel in selected:
            self._population.discard(sel[1])

    def grow_by_nr_of_generations(self, nr_of_generaitons: int, eval_data: EvaluationData):
        
        logging.info("Measuring fitness of the initial population...");
        self._population = SortedList(map(lambda x: EvaluatedIndividual(x, eval_data), self._population_raw))
        logging.info("Measuring initial population done! individuals: %f, best's fitness: %f", len(self._population), self._population[-1].fitness)

        for i in range(nr_of_generaitons):

            logging.info("Growing generation %d...", i)
            self.reproduce(eval_data)
            self.replace()
            logging.info("Growing done for generation %d! individuals: %d, best's fitness: %d", i, len(self._population), self._population[-1].fitness)

    @property
    def population(self):
        return self._population

    @staticmethod
    def from_blueprint(ann_blueprint: ArtificialNn, lambda_list):
        population = [lbd(copy(ann_blueprint)).compile() for lbd in lambda_list]
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
                [lambda x: x.with_batch_size(ind["batchSize"])
                            .with_epochs(ind["epochs"])
                            .with_layers(
                                [LayerWrapper(layer["size"], layer["activation"]) 
                                for layer in ind["layers"]]
                            )
                for ind in config["individuals"]]
            )

