from KerasWrapper.Problems.ProblemBase import ProblemBase
from typing import List

class HeartDesease(ProblemBase):

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
