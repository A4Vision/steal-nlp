import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "hw3", "data")
sys.path.insert(0, ROOT_DIR)
from hw3 import model_training_utils
from hw3 import inputs_generator


class TestBatchIterator(unittest.TestCase):
    def test_iterate_naive(self):
        g1 = inputs_generator.SubsetInputsGenerator([['a', 'b']])
        g2 = inputs_generator.SubsetInputsGenerator([['a', 'b'], ['c', 'd']], shuffle=False)
        iterator = model_training_utils.BatchDataIterator(g2, g1)
        total = 0
        for i, batch in enumerate(iterator.generate_data([4, 2, 2, 6])):
            total += sum(map(len, batch))
            self.assertEqual(iterator.nqueries(), total)
            if i == 0:
                self.assertListEqual(batch, [['a', 'b'], ['a', 'b']])
                self.assertEqual(iterator.n_unique_words(), 2)
            elif i == 1:
                self.assertListEqual(batch, [['a', 'b']])
                self.assertEqual(iterator.n_unique_words(), 2)
            elif i == 2:
                self.assertListEqual(batch, [['c', 'd']])
                self.assertEqual(iterator.n_unique_words(), 4)
            elif i == 3:
                self.assertListEqual(batch, [['a', 'b'], ['c', 'd'], ['a', 'b']])
                self.assertEqual(iterator.n_unique_words(), 4)
