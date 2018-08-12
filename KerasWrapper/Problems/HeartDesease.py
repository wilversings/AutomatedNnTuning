from enum import Enum
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from KerasWrapper.Problems.ProblemBase import ProblemBase, ProblemType
from typing import List
import numpy as np

"""
P.2
"""
class HeartDesease(ProblemBase):

    INPUT_SIZE = 26
    OUTPUT_SIZE = 5
    PROB_TYPE = ProblemType.Classification

    class Attributes:
        AGE = 0
        SEX = 1
        CP = 2
        TRESTBPS = 3
        CHOL = 4
        FBS = 5
        RESTECG = 6
        THALACH = 7
        EXANG = 8
        OLDPEAK = 9
        SLOPE = 10
        CA = 11
        THAL = 12
        NUM = 13

    def __init__(self, uri):
        super().__init__(uri)
        self._logger.info("Starting new population to test: P.2")

    def _load_data(self, uri):
        with open(uri) as inp:
            dataset = np.array([
                [float(x) if x.strip() != '?' else 1 for x in rec.split(',')] for rec in inp
            ])

        output = np.array(dataset[:,HeartDesease.Attributes.NUM], dtype='int')
        dataset = np.delete(dataset, HeartDesease.Attributes.NUM, axis=1)

        dataset = HeartDesease.normalize(dataset, HeartDesease.Attributes.THAL)
        dataset = HeartDesease.normalize(dataset, HeartDesease.Attributes.SLOPE)
        dataset = HeartDesease.normalize(dataset, HeartDesease.Attributes.EXANG)
        dataset = HeartDesease.normalize(dataset, HeartDesease.Attributes.RESTECG)
        dataset = HeartDesease.normalize(dataset, HeartDesease.Attributes.FBS)
        dataset = HeartDesease.normalize(dataset, HeartDesease.Attributes.CP)
        dataset = HeartDesease.normalize(dataset, HeartDesease.Attributes.SEX)

        return list(zip(dataset, [HeartDesease.val_to_out_layer(x) for x in output]))

    @staticmethod
    def val_to_out_layer(val: int) -> List[int]:
        ans = [0] * 5
        ans[val] = 1
        return ans

    @staticmethod
    def normalize(dataset, ind):
        enc = OneHotEncoder()
        le = LabelEncoder()

        one_hot = enc.fit_transform(le.fit_transform(dataset[:,ind]).reshape(-1, 1))
        one_hot = np.array(one_hot.todense())
        dataset = np.delete(dataset, ind, axis=1)
        return np.concatenate((dataset, one_hot), axis=1)
