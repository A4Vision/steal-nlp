import os
import sys
import unittest


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "hw3", "data")
sys.path.insert(0, ROOT_DIR)
from hw3 import model_interface
from hw3 import model_training_utils
from hw3 import data_loader
from hw3 import memm
from hw3.classifiers import sparse_logistic_regression


class TestDataAccumulator(unittest.TestCase):
    def test_consistent_with_model(self):
        vectorizer = memm.get_dict_vectorizer(DATA_PATH, None, 20)
        logistic = sparse_logistic_regression.SparseBinaryLogisticRegression.load(
            os.path.join(DATA_PATH, "all_freq20_my.pkl"))
        interface = model_interface.ModelInterface(logistic, vectorizer)
        accumulator = model_training_utils.TrainingDataAccumulator(vectorizer, interface)
        self.assertIsNone(accumulator.labels())
        self.assertIsNone(accumulator.indices())
        data = data_loader.DataPreprocessor(DATA_PATH, 20, 2000, False)
        sentences = data.original_preprocessed_data('dev')
        accumulator.query(sentences)
        indices = accumulator.indices()
        labels = accumulator.labels()
        accumulator.query(sentences)
        new_indices = accumulator.indices()[indices.shape[0]:]
        self.assertEqual(indices.shape, new_indices.shape)
        indices_equal = all([(indices[i] == new_indices[i]).all() for i in xrange(len(indices))])
        new_labels = accumulator.labels()[indices.shape[0]:]
        self.assertTrue(indices_equal)
        self.assertTrue((new_labels == labels).all())
        self.assertTrue((logistic.predict(indices) == labels).all())


