from collections import defaultdict
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

        a = [-1, 1]

        r_compress = srows > trows
        r_ls = Utils.linspace(srows + a[r_compress], trows + r_compress)

        c_compress = scols > tcols
        c_ls = Utils.linspace(scols + a[c_compress], tcols + c_compress)

        r_ls_rev = defaultdict(lambda: 0)
        c_ls_rev = defaultdict(lambda: 0)
        for el in r_ls: r_ls_rev[el] += 1
        for el in c_ls: c_ls_rev[el] += 1

        ans = [[None] * tcols for _ in range(trows)]
        for rind in range(r_compress, trows + r_compress):
            for cind in range(c_compress, tcols + c_compress):

                if r_compress and c_compress:
                    cell = np.mean(mat[r_ls[rind - 1]: r_ls[rind], c_ls[cind - 1]: c_ls[cind]])
                elif not r_compress and c_compress:
                    cell = np.mean(mat[r_ls[rind], c_ls[cind - 1]: c_ls[cind]]) / r_ls_rev[r_ls[rind]]
                elif r_compress and not c_compress:
                    cell = np.mean(mat[r_ls[rind - 1]: r_ls[rind], c_ls[cind]]) / c_ls_rev[c_ls[cind]]
                else:
                    cell = mat[r_ls[rind], c_ls[cind]] / (r_ls_rev[r_ls[rind]] * c_ls_rev[c_ls[cind]])

                ans[rind - r_compress][cind - c_compress] = cell

        return np.array(ans)