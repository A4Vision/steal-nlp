"""
Measure the runtime of various input generation methods.
"""
import numpy as np
import os
import argparse
import scipy.sparse
import sys
import numpy as np
import time
import scipy.stats

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(BASE_DIR, "data")
sys.path.insert(0, ROOT_DIR)
from hw3 import inputs_generator
from hw3 import memm
from hw3 import utils
from hw3 import model_interface
from hw3.classifiers import sparse_logistic_regression
import ngram_model


def measure_generator(generator, nqueries, description):
    t = utils.Timer("select from {} generates {:.1f}K queries".format(description, nqueries / 1000.))
    s = generator.generate_many_inputs(nqueries)
    print 'Average likelihood', ngram.sentences_perplexity(s)
    print t


def measure_ngram_model(nqueries):
    global ngram
    ngram_generator = inputs_generator.NGramModelGenerator(20, ngram)
    measure_generator(ngram_generator, nqueries, "ngram")


def test_score_whole_sentence():
    global dict_vectorizer, original_model_interface
    print "testing whole sentence score"
    scorer = inputs_generator.MaxEntropy(dict_vectorizer, model, original_model_interface)
    for sentence in untagged_train_sents[:100]:
        # Specialized efficient implementation
        s1 = scorer.score_whole_sentence(sentence)
        # naive implementation - sum scores
        s2 = inputs_generator.InputScorer.score_whole_sentence(scorer, sentence)
        assert abs(s1 - s2) < 1e-5
    print "SUCCESS"


def measure_select_from_ngram(nqueries):
    global ngram, dict_vectorizer, original_model_interface
    amount = 10
    ngram_generator = inputs_generator.NGramModelGenerator(20, ngram)
    scorer = inputs_generator.MaxEntropy(dict_vectorizer, model, original_model_interface)
    select_generator = inputs_generator.SelectFromOtherInputsGenerator(ngram_generator, scorer, amount )
    measure_generator(select_generator, nqueries, "select {} from ngram".format(amount))


def create_globals():
    global ngram_model, model, original_model_interface, ngram, validation_indices, dict_vectorizer, untagged_train_sents

    dict_vectorizer = memm.get_dict_vectorizer(DATA_PATH, None, 20)
    model = sparse_logistic_regression.SparseBinaryLogisticRegression.load(os.path.join(DATA_PATH, "all_freq20_my.pkl"))
    original_model_interface = model_interface.ModelInterface(model, dict_vectorizer)
    train_sents, dev_sents, test_sents = memm.load_train_dev_test_sentences(DATA_PATH, 20)

    untagged_train_sents = map(memm.untag_sentence, train_sents)

    print "generating sparse features examples - validation"
    validation_list_of_all_probs, tagged_validation_sents = utils.experiment1_generate_training_examples(
        dev_sents[:200], original_model_interface)

    print "creating input for training"
    validation_probs, validation_sparse_features, validation_predictions = \
        memm.transform_input_for_training(dict_vectorizer, validation_list_of_all_probs,
                                          tagged_validation_sents)
    # TODO(bugabuga): create validation indices when applying a good preprocessing - replacing more words.
    validation_indices = utils.csr_indices(validation_sparse_features)

    ngram = ngram_model.NGramModel(untagged_train_sents, 0.3, 0.6)


def measure_beam_search(nqueries):
    scorer = inputs_generator.MaxEntropy(dict_vectorizer, model, original_model_interface)
    print ngram.sentences_perplexity([ngram.generate_sentence(30)])
    beam = inputs_generator.BeamSearchInputGenerator(ngram, scorer, 100, 3, 2, 20, 30)
    measure_generator(beam, nqueries, "beam search")


if __name__ == '__main__':
    create_globals()
    measure_beam_search(1000)
    test_score_whole_sentence()
    measure_ngram_model(100 * 1000)
    measure_select_from_ngram(1000)

