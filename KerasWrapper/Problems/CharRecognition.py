from KerasWrapper.Problems.ProblemBase import ProblemBase
import numpy as np
from typing import List

class CharRecognition(ProblemBase):
    
    def __init__(self, uri_train, uri_train_labels, uri_test, uri_test_labels):
        super(CharRecognition, self).__init__((uri_train, uri_train_labels, uri_test, uri_test_labels))

    def _load_data(self, uri):
        uri_train, uri_train_labels, uri_test, uri_test_labels = uri

        train_in, train_out = CharRecognition.parse(uri_train, uri_train_labels, 60000)
        test_in, test_out = CharRecognition.parse(uri_test, uri_test_labels, 10000)

        return list(zip(np.concatenate((train_in, test_in)), np.concatenate((train_out, test_out))))

    @staticmethod
    def parse(uri, uri_labels, batch_size):
        with open(uri, 'rb') as mnist:
            mnist.seek(0x10)
            bytes = np.ndarray.astype(np.array(bytearray(mnist.read())), 'int16')

        with open(uri_labels, 'rb') as labels:
            labels.seek(8)
            label = bytearray(labels.read())

        mnist = np.array_split(bytes, batch_size)

        return np.array(mnist) / 255, np.array(list(map(CharRecognition.label_to_out_layer, label)))
    
    @staticmethod
    def label_to_out_layer(label: int) -> List[int]:
        ans = [0] * 10
        ans[label] = 1
        return ans