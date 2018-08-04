from KerasWrapper.Problems.ProblemBase import ProblemBase, ProblemType
from typing import List

"""
P.2
"""
class HeartDesease(ProblemBase):

    INPUT_SIZE = 13
    OUTPUT_SIZE = 5
    PROB_TYPE = ProblemType.Classification

    def __init__(self, uri):
        super().__init__(uri)
        self._logger.info("Starting new population to test: P.2")

    def _load_data(self, uri):
        with open(uri) as inp:

            ans = []
            for rec in inp:
                nr = [float(x) if x.strip() != '?' else -1 for x in rec.split(',')]
                ans.append( (nr[:-1], HeartDesease.val_to_out_layer(int(nr[-1]))) )
            return ans

    @staticmethod
    def val_to_out_layer(val: int) -> List[int]:
        ans = [0] * 5
        ans[val] = 1
        return ans
