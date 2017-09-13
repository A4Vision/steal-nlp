import argparse
import collections
import os
import sys
import time

import numpy as np


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(BASE_DIR, "data")
sys.path.insert(0, ROOT_DIR)
from hw3 import data_loader
from hw3 import model_training_utils
from hw3.classifiers import sparse_logistic_regression
from hw3 import inputs_generator
from hw3 import randomizers
from hw3 import utils
from hw3 import model_interface

# found "SEQUENTIAL", "IID_WORDS", to be inefficient.
STRATEGIES = ["UNIGRAMS", "MAX_SIGNIFICANCE", "FROM_TRAIN_SET", "MAX_GRADIENT", "MAX_ENTROPY",
              "FROM_TRAIN_SET_MAX_ENTROPY", "FROM_TRAIN_SET_MAX_GRADIENT",
              "LANGUAGE_MODEL", "MAX_ENTROPY_LANGUAGE_MODEL", "MAX_GRADIENT_LANGUAGE_MODEL",
              "BEAM_LANGUAGE_MODEL", "BEAM_MAX_ENTROPY_LANGUAGE_MODEL", "BEAM_MAX_GRADIENT_LANGUAGE_MODEL"]
END_MESSAGE = "Labels Only Experiment Done"


def experiment(stolen_model_fname, original_model_interface, data_preprocess, stolen_model,
               eta, l2_weight, loss_improvement, maximal_batch_time, batches_sizes,
               sentences_generator, first_senteneces_generator):
    assert isinstance(stolen_model, sparse_logistic_regression.SparseBinaryLogisticRegression)
    assert isinstance(sentences_generator, inputs_generator.InputGenerator)
    assert ".pkl" not in stolen_model_fname
    print "Loading data"
    dev_size = 800

    data_accumulator = model_training_utils.TrainingDataAccumulator(original_model_interface.dict_vectorizer(),
                                                                    original_model_interface)
    print "Generate validation data"
    dev_indices, dev_labels = data_accumulator.get_indices_and_labels(
        data_preprocess.original_preprocessed_data('dev')[:dev_size])
    dev_probs = original_model_interface.underlying_model().predict_proba(dev_indices)
    print "Creating input for training"
    statistics = model_training_utils.StatisticsLogger(dev_indices, dev_probs, dev_labels)

    data_iterator = model_training_utils.BatchDataIterator(sentences_generator, first_senteneces_generator)
    timer = utils.Timer("sentences generation")
    for new_sentences in data_iterator.generate_data(batches_sizes):
        timer.start_part("query")
        data_accumulator.query(new_sentences)
        training_losses = []
        start_time = time.time()
        timer.start_part("optimization")
        for i in xrange(1, 10000):
            current_eta = eta / i ** 0.1
            stolen_model.epoch_optimize(data_accumulator.indices(), data_accumulator.labels(), 100, current_eta,
                                        l2_weight)
            stolen_model.minimize_norm()
            if i % 10 == 0:
                print 'epoch=', i, 'eta=', current_eta
                accuracy = np.average(stolen_model.predict(dev_indices) == dev_labels)
                print 'validation accuracy', accuracy
                train_loss = stolen_model.loss(data_accumulator.indices(), data_accumulator.labels(), l2_weight)
                print 'training_loss', train_loss
                training_losses.append(train_loss)
                # Twenty minutes per optimization problem.
                if time.time() - start_time > maximal_batch_time:
                    print 'Training time exceeded, breaking'
                    break
                if len(training_losses) > 10 and train_loss * (1 + loss_improvement) > np.average(
                        training_losses[-10: -1]):
                    print 'Training loss stopped improving, breaking'
                    break

        print 'Optimization time: {}seconds'.format(time.time() - start_time)
        timer.start_part("log state")
        statistics.log_state(stolen_model, original_model_interface, l2_weight, train_loss, data_iterator.nqueries(),
                             data_iterator.n_unique_words())
        print 'current training losses'
        print training_losses
        timer.start_part("sentences generation")

    # stolen_model.save(os.path.join(DATA_PATH, "{}_queries{}.pkl".format(stolen_model_fname, single_word_queries_amount)))
    print timer
    statistics.print_logged_data()
    print END_MESSAGE


def count_set(sents):
    counter = collections.Counter()
    for s in sents:
        counter.update(s)
    return counter


def create_sentences_generators(args, stolen_model, length_generator,
                                # original_interface should be secret -
                                # we cheat subtly to test the methods' potential.
                                original_interface,
                                data_preprocess):
    # Load training data for statistics.
    train_sentences = data_preprocess.original_preprocessed_data('train')
    # Create naive generators.
    train_set_generator = inputs_generator.SubsetInputsGenerator(train_sentences)
    language_model_generator = inputs_generator.NGramModelGenerator(40, data_preprocess.narrow_ngram_language_model())
    iid_generator = inputs_generator.GreedyInputsGenerator(length_generator,
                                                           randomizers.RandomizeByFrequenciesIIDFromDict(
                                                               {w: 1 for w in data_preprocess.words()}),
                                                           inputs_generator.TrivialInputScorer(), 1)

    train_freq = count_set(train_sentences)
    train_freq_clipped = {w: train_freq[w] for w in data_preprocess.words()}
    proportional_words_randomizer = randomizers.RandomizeByFrequencyProportionaly(
        train_freq_clipped, 1.05)

    # Create scorers
    trivial_scorer = inputs_generator.TrivialInputScorer()
    # Cheating scorers
    # max_entropy_scorer = inputs_generator.MaxEntropy(data_preprocess.dict_vectorizer(), stolen_model,
    #                                                  original_interface)
    # subtle_decision_scorer = inputs_generator.SubtleDecision(data_preprocess.dict_vectorizer(), stolen_model,
    #                                                          original_interface)
    # max_gradient_scorer = inputs_generator.MaximalGradient(data_preprocess.dict_vectorizer(), stolen_model,
    #                                                        original_interface)
    # No cheat scorers
    # TODO(bugabuga): use the no cheat scorers instead of the cheating scorers !
    stolen_model_interface = model_interface.ModelInterface(stolen_model, data_preprocess.dict_vectorizer())
    honest_max_entropy_scorer = inputs_generator.MaxEntropy(data_preprocess.dict_vectorizer(), stolen_model,
                                                            stolen_model_interface)
    honest_subtle_decision_scorer = inputs_generator.SubtleDecision(data_preprocess.dict_vectorizer(), stolen_model,
                                                                    stolen_model_interface)
    honest_max_gradient_scorer = inputs_generator.MaximalGradient(data_preprocess.dict_vectorizer(), stolen_model,
                                                                  stolen_model_interface)

    # Switch(args.strategy) --> create sentences_generator according to the selected strategy.
    if args.strategy == "BEAM_LANGUAGE_MODEL":
        sentences_generator = inputs_generator.BeamSearchInputGenerator(data_preprocess.narrow_ngram_language_model(),
                                                                        trivial_scorer, 100, 3, 3, 20, 30,
                                                                        data_preprocess.original_ngram_language_model())
    elif args.strategy == "BEAM_MAX_ENTROPY_LANGUAGE_MODEL":
        sentences_generator = inputs_generator.BeamSearchInputGenerator(data_preprocess.narrow_ngram_language_model(),
                                                                        honest_max_entropy_scorer, 100, 3, 3, 20, 30,
                                                                        data_preprocess.original_ngram_language_model())
    elif args.strategy == "BEAM_MAX_GRADIENT_LANGUAGE_MODEL":
        sentences_generator = inputs_generator.BeamSearchInputGenerator(data_preprocess.narrow_ngram_language_model(),
                                                                        honest_max_gradient_scorer, 100, 3, 3, 20, 30,
                                                                        data_preprocess.original_ngram_language_model())
    elif args.strategy == "FROM_TRAIN_SET_MAX_ENTROPY":
        sentences_generator = inputs_generator.SelectFromOtherInputsGenerator(
            train_set_generator, honest_max_entropy_scorer, 10)
    elif args.strategy == "FROM_TRAIN_SET_MAX_GRADIENT":
        sentences_generator = inputs_generator.SelectFromOtherInputsGenerator(
            train_set_generator, honest_max_gradient_scorer, 10)
    elif args.strategy == "MAX_ENTROPY_LANGUAGE_MODEL":
        sentences_generator = inputs_generator.SelectFromOtherInputsGenerator(
            language_model_generator, honest_max_entropy_scorer, 10)
    elif args.strategy == "MAX_GRADIENT_LANGUAGE_MODEL":
        sentences_generator = inputs_generator.SelectFromOtherInputsGenerator(
            language_model_generator, honest_max_gradient_scorer, 10)
    elif args.strategy == "LANGUAGE_MODEL":
        sentences_generator = language_model_generator
    elif args.strategy == "UNIGRAMS":
        sentences_generator = inputs_generator.GreedyInputsGenerator(length_generator,
                                                                     proportional_words_randomizer,
                                                                     trivial_scorer, 1)
    elif args.strategy == "MAX_SIGNIFICANCE":
        sentences_generator = inputs_generator.GreedyInputsGenerator(length_generator,
                                                                     proportional_words_randomizer,
                                                                     honest_subtle_decision_scorer, 10)
    elif args.strategy == "MAX_GRADIENT":
        sentences_generator = inputs_generator.GreedyInputsGenerator(length_generator,
                                                                     proportional_words_randomizer,
                                                                     honest_max_gradient_scorer, 10)
    elif args.strategy == "MAX_ENTROPY":
        sentences_generator = inputs_generator.GreedyInputsGenerator(length_generator,
                                                                     proportional_words_randomizer,
                                                                     honest_max_entropy_scorer, 10)
    elif args.strategy == "FROM_TRAIN_SET":
        sentences_generator = train_set_generator
    elif args.strategy == "IID_WORDS":
        sentences_generator = iid_generator
    elif args.strategy == "SEQUENTIAL":
        sentences_generator = inputs_generator.SequentialInputsGenerator(length_generator, data_preprocess.words())
    else:
        assert args.strategy in STRATEGIES, args.strategy
        assert False, args.strategy
    return iid_generator, sentences_generator


def main():
    parser = argparse.ArgumentParser(description='Steal POS model.')
    parser.add_argument("--original_model_file_name", type=str, help="File name for the original classifier.",
                        required=True)
    parser.add_argument("--stolen_model_file_name", type=str, help="File name for the stolen classifier.",
                        required=True)
    parser.add_argument("--eta", type=float, help="Learning rate.", required=True)
    parser.add_argument("--l2_weight", type=float, help="L2 weight.", required=True)
    parser.add_argument("--loss_improvement", type=float, help="Maximal optimization runtime.", required=True)
    parser.add_argument("--minimal_frequency", type=int,
                        help="Minimal frequency for a word to be observed not unknown.", required=True)
    parser.add_argument("--total_queries_amount", type=int, help="Maximal amount of queries to use.", required=True)
    parser.add_argument("--batch_size", type=int, help="Number of queries per batch.", required=True)
    parser.add_argument("--strategy", choices=STRATEGIES,
                        type=str, help="Input sentences generation strategy.", required=True)
    parser.add_argument("--first_random", type=int,
                        help="Number of initial random queries.", required=True)
    parser.add_argument("--num_words", type=int, help="Number of words to use in the queries.", required=True)
    parser.add_argument("--max_batch_time_secs", type=float, help="Maximal SGD optimization time for each batch.",
                        required=True)

    try:
        args = parser.parse_args(sys.argv[1:])
    except:
        parser.print_help()
        raise
    assert os.path.sep not in args.original_model_file_name
    assert os.path.sep not in args.stolen_model_file_name

    print ' '.join(sys.argv)
    print args

    data_preprocess = data_loader.DataPreprocessor(DATA_PATH, args.minimal_frequency, args.num_words)
    original_model = sparse_logistic_regression.SparseBinaryLogisticRegression.load(
        os.path.join(DATA_PATH, args.original_model_file_name))
    original_interface = model_interface.ModelInterface(original_model, data_preprocess.dict_vectorizer())

    LENGTH = 25
    length_generator = inputs_generator.constant_generator(LENGTH)

    batches_sizes = [args.first_random] + [args.batch_size] * (
        (args.total_queries_amount - args.first_random) // args.batch_size)

    shape = original_model.w().shape
    stolen_model = sparse_logistic_regression.SparseBinaryLogisticRegression(10, shape[0], shape[1])
    first_senteneces_generator, sentences_generator = create_sentences_generators(args, stolen_model, length_generator,
                                                                                  original_interface,
                                                                                  data_preprocess)

    experiment(args.stolen_model_file_name, original_interface,
               data_preprocess, stolen_model,
               args.eta, args.l2_weight, args.loss_improvement, args.max_batch_time_secs,
               batches_sizes, sentences_generator, first_senteneces_generator)


if __name__ == '__main__':
    main()
