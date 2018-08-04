from KerasWrapper.Problems.ProblemBase import ProblemBase, ProblemType
from typing import List
import csv

"""
P.3
"""
class SportArticleObjectivity(ProblemBase):

    INPUT_SIZE = 51
    OUTPUT_SIZE = 2
    PROB_TYPE = ProblemType.BinaryClassification

    def __init__(self, uri):
        super().__init__(uri)
        self._logger.info("Starting new population to test: P.3")

    def _load_data(self, uri):
        with open(uri) as inp:
            return [([int(val) / int(line[3]) for val in line[4:-3]] + [float(line[-3]), float(line[-2])], self.val_to_out_layer(line[2])) for line in csv.reader(inp, delimiter=',', quotechar='"')]

    @staticmethod
    def val_to_out_layer(val: str) -> List[int]:
        if val == "objective":
            return [1, 0]
        elif val == "subjective":
            return [0, 1]
        assert False

