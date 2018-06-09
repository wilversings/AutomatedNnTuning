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