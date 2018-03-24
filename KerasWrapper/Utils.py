from random import choice, sample


class Utils(object):

    @staticmethod
    def flip_coin():
        return choice([True, False])

    @staticmethod
    def ordered_sample(list, k):
        indices = sample(range(len(list)), k)
        return [list[i] for i in sorted(indices)]