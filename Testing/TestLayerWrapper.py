import unittest
from KerasWrapper.Wrappers.LayerWrapper import LayerWrapper

class Test_TestLayerWrapper(unittest.TestCase):

    def test_crossover(self):

        l1 = LayerWrapper(2, "softmax")
        l2 = LayerWrapper(8, "softmax")

        son = l1.crossover(l2)

        self.assertEqual(son.size, 5)
        self.assertEqual(son.activation, "softmax")

if __name__ == '__main__':
    unittest.main()
