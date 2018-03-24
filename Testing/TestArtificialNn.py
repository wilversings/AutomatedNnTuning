import unittest
from KerasWrapper.Wrappers.LayerWrapper import LayerWrapper
from KerasWrapper.Wrappers.ArtificialNn import ArtificialNn

class Test_TestArtificialNn(unittest.TestCase):
    
    def test_crossover(self):

        nn1 = ArtificialNn(12, 10, True).with_epochs(50)\
        .with_batch_size(10)\
        .with_layers([
            LayerWrapper(5, 'relu'),
            LayerWrapper(10, 'relu'),
            LayerWrapper(24, 'relu')
        ]);

        nn2 = ArtificialNn(12, 10, True).with_epochs(30)\
        .with_batch_size(5)\
        .with_layers([
            LayerWrapper(11, 'relu'),
            LayerWrapper(2, 'relu'),
            LayerWrapper(1, 'whatever')
        ])

        child = nn1.crossover(nn2)

        # Delta 1 due to possible mutations
        self.assertAlmostEqual(first=child.layers[0].size, second=8, delta=1)
        self.assertAlmostEqual(first=child.layers[1].size, second=6, delta=1)
        self.assertAlmostEqual(first=child.layers[2].size, second=12, delta=1)


if __name__ == '__main__':
    unittest.main()
