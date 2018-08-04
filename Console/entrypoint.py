import logging

from KerasWrapper.Problems.Bulldozers import Bulldozers
from KerasWrapper.Problems.SportArticleObjectivity import SportArticleObjectivity
from KerasWrapper.Wrappers.EvaluationData import EvaluationData
from KerasWrapper.Tuners.Population import Population
from KerasWrapper.Problems.CharRecognition import CharRecognition
from KerasWrapper.Problems.HeartDesease import HeartDesease
import os
import time

if __name__ == '__main__':
    print ("You are here: " + os.getcwd())
    print('\n')

    Population.NAME = input("Name of your population > ")

    if not os.path.exists(Population.NAME):
        os.makedirs(Population.NAME)
    else:
        print("Population already exists")
        exit(-1)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename= Population.NAME + "/general.log",
                        filemode='w')

    # Problem = Bulldozers
    # problem = Bulldozers('TrainAndValid.csv')
    # test_in, test_out, train_in, train_out = problem.perform_k_fold(150)
    # train_in = train_in[:850]
    # train_out = train_out[:850]

    # Problem = CharRecognition
    # problem = CharRecognition('train/mnist', 'train/mnist_labels', 'test/mnist_test', 'test/mnist_test_labels') # input = 784, output = 10
    # test_in, test_out, train_in, train_out = problem.perform_k_fold(150)
    # train_in = train_in[:850]
    # train_out = train_out[:850]

    # Problem = SportArticleObjectivity
    # problem = SportArticleObjectivity('features.csv') # input = 57, output = 2
    # test_in, test_out, train_in, train_out = problem.perform_k_fold(150)

    Problem = HeartDesease
    problem = HeartDesease('heartdesease.data') # input = 13, output = 5
    test_in, test_out, train_in, train_out = problem.perform_k_fold(50)

    print("Training entries: " + str(len(train_in)) + " Testing entries: " + str(len(test_in)))
    pop = Population.generate_rand_population(pop_size=         50,
                                              input_size=       Problem.INPUT_SIZE,
                                              output_size=      Problem.OUTPUT_SIZE,
                                              layer_nr_range=   (2,8),
                                              layer_size_range= (10, 40),
                                              batch_size=       20,
                                              epochs=           10,
                                              prob_type=        Problem.PROB_TYPE)
    eval_data = EvaluationData(test_in, test_out, train_in, train_out)

    pop.grow_by_nr_of_generations(100, eval_data)
