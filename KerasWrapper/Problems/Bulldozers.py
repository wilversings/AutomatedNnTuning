from KerasWrapper.Problems.ProblemBase import ProblemBase
import numpy as np

"""
P.4
"""
class Bulldozers(ProblemBase):


    def __init__(self, uri):
        super().__init__(uri)
        self._logger.info("Starting new population to test: P.4")

    # def _load_data(self, uri):
    #     with open(uri) as inp:
    #         return [([int(val) / int(line[3]) for val in line[4:-3]] + [float(line[-3]), float(line[-2])], self.val_to_out_layer(line[2])) for line in csv.reader(inp, delimiter=',', quotechar='"')]
    #
    # @staticmethod
    # def val_to_out_layer(val: str) -> List[int]:
    #     if val == "objective":
    #         return [1, 0]
    #     elif val == "subjective":
    #         return [0, 1]
    #     assert False