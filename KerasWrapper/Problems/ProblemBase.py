from random import shuffle
from abc import abstractmethod
from abc import ABC

class ProblemBase(ABC):
    
    @abstractmethod
    def _load_data(self, uri): pass

    def __init__(self, uri):
        self._database = self._load_data(uri);

    def perform_k_fold(self, k):

        db_copy = self._database[:]
        shuffle(db_copy)

        return db_copy[:k], db_copy[k:]

