from random import choice, sample, random
from typing import List
import numpy as np


class Utils(object):

    @staticmethod
    def flip_coin() -> bool:
        return choice([True, False])

    @staticmethod
    def uneven(chance: float) -> bool:
        return random() < chance

    @staticmethod
    def linspace(n: int, k: int):
        return np.linspace(0, n, k, dtype=int)

    @staticmethod
    def ordered_sample(list: List, k: int) -> List:
        indices = sample(range(len(list)), k)
        return [list[i] for i in sorted(indices)]

    @staticmethod
    def rebin(mat, target_shape) -> List:
        srows, scols = mat.shape
        trows, tcols = target_shape

        r_compress = srows > trows
        r_ls = Utils.linspace(srows + 1, trows + 1)

        c_compress = scols > tcols
        c_ls = Utils.linspace(scols + 1, tcols + 1)

        ans = [[None] * tcols for _ in range(trows)]
        if r_compress and c_compress:
            for rind in range(1, trows + 1):
                for cind in range(1, tcols + 1):
                    ans[rind - 1][cind - 1] = np.mean(mat[r_ls[rind - 1]: r_ls[rind], c_ls[cind - 1]: c_ls[cind]])


        return ans


mat = [
    [1, 5, 3, 2],
    [6, 23, 1, 43],
    [45, 7, 34, 53],
    [45, 1, 5, 7],
    [7, 5, 1, 2]
]
print(Utils.rebin(np.array(mat), (3, 3)))