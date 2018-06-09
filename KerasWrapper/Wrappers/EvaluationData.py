import numpy as np


class EvaluationData:

    def __init__(self, 
                 test_in:   list,
                 test_out:  list,
                 train_in:  list, 
                 train_out: list):

        self._test_in = test_in
        self._test_out = test_out
        self._train_in = train_in
        self._train_out = train_out
        
    @property
    def test_in(self):
        return self._test_in
    @property
    def test_out(self):
        return self._test_out
    @property
    def train_in(self):
        return self._train_in
    @property
    def train_out(self):
        return self._train_out


