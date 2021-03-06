from random import shuffle
from abc import abstractmethod
from abc import ABC
import numpy as np
import logging
from enum import Enum

class ProblemType(Enum):
    Classification = 1
    BinaryClassification = 2
    Regression = 3

class ProblemBase(ABC):
    
    @abstractmethod
    def _load_data(self, uri): pass

    def __init__(self, uri):
        self._database = self._load_data(uri);
        self._logger = logging.getLogger("problem")

    def perform_k_fold(self, k):

        db_copy = self._database[:]
        shuffle(db_copy)

        test_in, test_out = list(zip(*db_copy[:k]))
        train_in, train_out = list(zip(*db_copy[k:]))
        return np.array(test_in), np.array(test_out), np.array(train_in), np.array(train_out)