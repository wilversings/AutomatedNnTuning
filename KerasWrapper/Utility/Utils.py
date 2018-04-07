from random import choice, sample, random


class Utils(object):

    @staticmethod
    def flip_coin():
        return choice([True, False])

    @staticmethod
    def uneven(chance):
        return random() < chance

    @staticmethod
    def ordered_sample(list, k):
        indices = sample(range(len(list)), k)
        return [list[i] for i in sorted(indices)]