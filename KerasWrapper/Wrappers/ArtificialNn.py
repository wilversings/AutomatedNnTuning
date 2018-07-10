from KerasWrapper.Wrappers.NeuralNetWrapper import NeuralNetWrapper
from KerasWrapper.Wrappers.EvaluationData import EvaluationData
from KerasWrapper.Utility.Utils import Utils
from random import choice, random, sample
from KerasWrapper.Evolutionary.Individual import Individual
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers.core import Dense
import logging


class ArtificialNn(NeuralNetWrapper, Individual):

    MUTATION_CHANCE = 0.2
    
    def __init__(self, input_size: int, output_size: int):
        
        NeuralNetWrapper.__init__(self, input_size, output_size)
        Individual.__init__(self)

    def compile(self):
        """
        Compiles the ArtificialNn decorator into a real Keras object,
        with regard about the input, hidden & output layers
        :return: Returns self for chainability reasons
        """

        k_layers = [
            Dense(
                units=      self._layers[0].size, 
                input_dim=  self._input_size,
                #weights=    [self.layers[0].init_weights, self.layers[0].init_biases]
        )] + [Dense(
                units=      x.size, 
                activation= x.activation,
                #weights=    [x.init_weights, x.init_biases]
             )  for x in self._layers[1:]
        ] + [Dense(
                units=      self._output_size
            )]

        model = Sequential()
        
        for layer in k_layers:
            model.add(layer)

        # Use softmax only for classification problems
        model.add(Activation("softmax"))

        model.compile(loss='categorical_crossentropy',
                #TODO: make optimizer a gene
              optimizer='adam',
              metrics=['accuracy'])

        return model

    def crossover(self, other: 'ArtificialNn') -> 'ArtificialNn':
        """
        Performs a crossover of the current Neural Network with another one

        :param other: the couterpart neural network
        :return: A new neural network
        """
                
        if __debug__:
            assert(self._input_size == other._input_size)
            assert(self._output_size == other._output_size)

        self_layers = self._layers
        other_layers = other._layers

        if len(self_layers) > len(other_layers):
            # Make sure that self_layers has less layers than ctp_layers
            self_layers, other_layers = other_layers, self_layers

        # Chose random samples from the net with more layers
        # other_layers = Utils.ordered_sample(other_layers, len(self_layers))
        ans_layer_nr = (len(other_layers) + len(self_layers)) // 2

        linsamples = Utils.linspace(len(other_layers), ans_layer_nr)
        rev_linsamples = Utils.linspace(len(self_layers) - 1, ans_layer_nr)

        new_layers = []
        trail = linsamples[0]
        for sample, rev_sample in list(zip(linsamples, rev_linsamples))[1:]:
            new_layers.append(self_layers[rev_sample].crossover(other_layers[trail:sample]).mutate())
            trail = sample

        return ArtificialNn(self._input_size, self._output_size)\
            .with_layers(
                new_layers
            )\
            .with_batch_size(
                # Chose one random batch size from the two parts
                (self._batch_size + other._batch_size) // 2
            )\
            .with_epochs(
                (self._epochs + other._epochs) // 2
            )

    def mutate(self):
        if random() < self.MUTATION_CHANCE:
            self._epochs = self._epochs + choice([-1, 1])
            self._batch_size = self._batch_size + choice([-1, 1])
        return self

    def measure_fitness(self, data: EvaluationData) -> float:

        model = self.compile()
        model.fit(data.train_in, data.train_out,
                           epochs=self._epochs, batch_size=self._batch_size, verbose=0)

        loss_and_metrics = model.evaluate(data.test_in, data.test_out, verbose=2)
        return loss_and_metrics[1]
