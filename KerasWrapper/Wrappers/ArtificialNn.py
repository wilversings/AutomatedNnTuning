from KerasWrapper.Wrappers.NeuralNetWrapper import NeuralNetWrapper
from KerasWrapper.Utils import Utils
from random import choice, random, sample
from KerasWrapper.Evolutionary.Individual import Individual
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers.core import Dense

class ArtificialNn(NeuralNetWrapper, Individual):

    MUTATION_CHANCE = 0.2

    def __init__(self, input_size, output_size, clasf_prob: bool):
        
        super(ArtificialNn, self).__init__(input_size, output_size)

        self.__clasf_prob = clasf_prob
        self.__k_model = None

    def compile(self):

        k_layers = [
            Dense(
                units=          self._layers[0].size, 
                input_dim=      self._input_size)
        ] + [Dense(
                units=      x.size, 
                activation= x.activation) for x in self._layers[1:]
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

    def crossover(self, other):
                
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
                list(map(lambda x, y: x.crossover(y).mutate(), other_layers, self_layers))
            )\
            .with_batch_size(
                # Chose one random batch size from the two parts
                # TODO: try with average
                choice([self._batch_size, other._batch_size])
            )\
            .with_epochs(
                choice([self._epochs, other._epochs])
            )

    def mutate(self):
        if random() < self.MUTATION_CHANCE:
            self._epochs = self._epochs + choice([-1, 1])
            self._batch_size = self._batch_size + choice([-1, 1])


    def measure_fitness(self, train_in, train_out, test_in, test_out):
        
        if __debug__:
            assert(self.__k_model is not None)

        self.__k_model.fit(train_in, train_out,
                           epochs=self._epochs, batch_size=self._batch_size, verbose=2)

        loss_and_metrics = self.__k_model.evaluate(test_in, test_out, verbose=2)
        return loss_and_metrics[1]