import AnnWrapper
from Individual import Individual
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers.core import Dense

class ArtificialNn(AnnWrapper, Individual):

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

        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

        self.__k_model = model

    def crossover(self, other):
        pass

    def mutate(self):
        pass
