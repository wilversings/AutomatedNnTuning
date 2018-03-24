from KerasWrapper.Wrappers.NeuralNetWrapper import NeuralNetWrapper
from random import choice, random
from KerasWrapper.Evolutionary.Individual import Individual
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers.core import Dense

class ArtificialNn(NeuralNetWrapper, Individual):

    MUTATION_CHANCE = 0.2

    def __init__(self, input_size, output_size, clasf_prob: bool):
        
        super(input_size, output_size)

        self.__clasf_prob = clasf_prob
        self.__k_model = None

    def compile(self):

        k_layers = [
            Dense(
                units=          self._layers[1].size, 
                input_size=     self._layers[0].size)
        ] + [Dense(
                units=      x.size, 
                activation= x.activation) for x in self._layers[2:]]

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

    def crossover(self, other):
                
        self_layers = self._layers
        other_layers = other._layers

        if len(self_layers) > len(other_layers):
            # Make sure that self_layers has less layers than ctp_layers
            self_layers, other_layers = other_layers, self_layers

        # Chose random samples from the net with more layers
        other_layers = random.sample(other_layers, len(self_layers))

        return NeuralNetWrapper(self._input_size, self._output_size)\
            .with_layers(
                # Crossover the choices
                list(map(lambda x, y: x.crossover(y).mutate(), other_layers, self_layers))
            )\
            .with_batch_size(
                # Chose one random batch size from the two parts
                # TODO: try with average
                random.choice([self._batch_size, other._batch_size])
            )\
            .with_epochs(
                random.choice([self._epochs, other._epochs])
            )

    def mutate(self):
        if random() < self.MUTATION_CHANCE:
            self._epochs = self._epochs + (-1 if (Utils.flip_coin() and self._epochs > 1) else 1)
            self._batch_size = self._batch_size + (-1 if (Utils.flip_coin() and self._batch_size > 1) else 1)


    def measure_fitness(self):
        pass