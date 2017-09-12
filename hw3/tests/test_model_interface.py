import os
import sys
import unittest
import numpy as np
import theanets

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, "..", '..'))
DATA_PATH = os.path.join(BASE_DIR, "data")
sys.path.insert(0, ROOT_DIR)

from hw3.classifiers import sparse_logistic_regression
from hw3 import convert_theanets_model
from hw3 import model_interface
from hw3 import memm
from hw3 import data
from hw3 import utils


class TestModelInterfaceCaching(unittest.TestCase):
    def test_caching_consistent_with_slow_prediction(self):
        train, dev, test = memm.load_train_dev_test_sentences(DATA_PATH, 20)
        theanets_model = theanets.Classifier.load(os.path.join(DATA_PATH, "all_freq20.pkl"))
        my_model = convert_theanets_model.convert_model(theanets_model)
        vectorizer = memm.get_dict_vectorizer(DATA_PATH, None, 20)
        fast_interface = model_interface.ModelInterface(my_model, vectorizer)
        slow_interface = model_interface.ModelInterface(theanets_model, vectorizer)
        for s in train[:100]:
            s = data.untag(s)
            probs_slow, tagged_slow = slow_interface.predict_proba(s)
            probs_fast1, tagged_fast1 = fast_interface.predict_proba_fast(s)
            probs_fast2, tagged_fast2 = fast_interface.predict_proba_fast(s)
            self.assertListEqual(tagged_fast1, tagged_slow)
            self.assertListEqual(tagged_fast1, tagged_fast2)
            self.assertLessEqual(np.abs(probs_slow - probs_fast1).max(), 1e-6)
            self.assertLessEqual(np.abs(probs_slow - probs_fast2).max(), 1e-6)

    def test_compare_times(self):
        train, dev, test = memm.load_train_dev_test_sentences(DATA_PATH, 20)
        theanets_model = theanets.Classifier.load(os.path.join(DATA_PATH, "all_freq20.pkl"))
        my_model = convert_theanets_model.convert_model(theanets_model)
        vectorizer = memm.get_dict_vectorizer(DATA_PATH, None, 20)
        fast_interface = model_interface.ModelInterface(my_model, vectorizer)
        slow_interface = model_interface.ModelInterface(theanets_model, vectorizer)
        N = 25
        t = utils.Timer("global")
        for s in train[:100]:
            s = data.untag(s)
            t.start_part("slow")
            for i in xrange(1, N):
                probs_slow, tagged_slow = slow_interface.predict_proba(s[:i])
            t.start_part("fast")
            print 'len(s)', len(s)
            for i in xrange(1, N):
                probs_fast2, tagged_fast2 = fast_interface.predict_proba_fast(s[:i])
        print t