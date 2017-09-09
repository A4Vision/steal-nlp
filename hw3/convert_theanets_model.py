import os
import sys
import theanets

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(BASE_DIR, "data")
sys.path.insert(0, ROOT_DIR)

from hw3.classifiers import sparse_logistic_regression
from hw3 import utils


def convert_model(theanets_classifier):
    w = theanets_classifier.params[0].get_value()
    b = theanets_classifier.params[1].get_value()
    assert len(theanets_classifier.params) == 2
    assert b.shape[0] == w.shape[1]
    res = sparse_logistic_regression.SparseBinaryLogisticRegression(10, w.shape[0], w.shape[1])
    res.set_b(b)
    res.set_w(utils.minimize_rows_norm(w))
    return res


def main():
    for src_model_fname in sys.argv[1:]:
        assert src_model_fname.endswith(".pkl")
        theanets_model = theanets.Classifier.load(src_model_fname)
        my_model = convert_model(theanets_model)
        my_model.save(src_model_fname[:-4] + "_my.pkl")


if __name__ == '__main__':
    main()