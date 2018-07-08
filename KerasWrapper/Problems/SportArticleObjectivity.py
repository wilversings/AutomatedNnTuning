from KerasWrapper.Problems.ProblemBase import ProblemBase
from typing import List
import csv

class SportArticleObjectivity(ProblemBase):

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

