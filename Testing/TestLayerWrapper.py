import unittest
from KerasWrapper.Wrappers.LayerWrapper import LayerWrapper


class Test_TestLayerWrapper(unittest.TestCase):

    def test_crossover(self):

        l1 = LayerWrapper(2, "softmax")
        l2 = LayerWrapper(8, "softmax")

        child = l1.crossover(l2)

        self.assertEqual(child.size, 5)
        self.assertEqual(child.activation, "softmax")

        l3 = LayerWrapper(2, "softmax")
        grandchild = l3.crossover(child)
        self.assertEqual(grandchild.size, 3)
        self.assertEqual(grandchild.activation, "softmax")


if __name__ == '__main__':
    unittest.main()
