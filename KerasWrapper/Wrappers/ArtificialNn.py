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

    def __init__(self, input_size: int, output_size: int, clasf_prob: bool):
        
        NeuralNetWrapper.__init__(self, input_size, output_size)
        Individual.__init__(self)

        self.__clasf_prob = clasf_prob
        self.__k_model = None

    def compile(self) -> 'ArtificialNn':
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
              
        if self.__clasf_prob:
            model.add(Activation("softmax"))
        else:
            # TODO: this
            pass

        model.compile(loss='categorical_crossentropy',
                #TODO: make optimizer a gene
              optimizer='adam',
              metrics=['accuracy'])

        self.__k_model = model

        return self

    def crossover(self, other: 'ArtificialNn') -> 'ArtificialNn':
        """
        Performs a crossover of the current Neural Network with another one

        :param other: the couterpart neural network
        :return: A new neural network
        """
                
        if __debug__:
            assert(self._input_size == other._input_size)
            assert(self._output_size == other._output_size)
            assert(self.__clasf_prob == other.__clasf_prob)

        self_layers = self._layers
        other_layers = other._layers

        if len(self_layers) > len(other_layers):
            # Make sure that self_layers has less layers than ctp_layers
            self_layers, other_layers = other_layers, self_layers

        # Chose random samples from the net with more layers
        other_layers = Utils.ordered_sample(other_layers, len(self_layers))

        return ArtificialNn(self._input_size, self._output_size, self.__clasf_prob)\
            .with_layers(
                # Crossover the choices
                [x.crossover(y).mutate() for x, y in zip(other_layers, self_layers)]
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
        if __debug__:
            assert(self.__k_model is not None)

        self.__k_model.fit(data.train_in, data.train_out,
                           epochs=self._epochs, batch_size=self._batch_size, verbose=2)

        loss_and_metrics = self.__k_model.evaluate(data.test_in, data.test_out, verbose=2)
        return loss_and_metrics[1]