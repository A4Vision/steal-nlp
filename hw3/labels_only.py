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

from hw3.classifiers import sparse_logistic_regression
from hw3 import memm
from hw3 import inputs_generator
from hw3 import randomizers
from hw3 import data
from hw3 import utils
from hw3 import model_interface

# found "SEQUENTIAL", "IID_WORDS", to be inefficient.
STRATEGIES = ["UNIGRAMS", "MAX_SIGNIFICANCE", "FROM_TRAIN_SET", "MAX_GRADIENT", "MAX_ENTROPY",
              "MAX_ENTROPY_FROM_TRAIN_SET", "MAX_GRADIENT_FROM_TRAIN_SET",
              "LANGUAGE_MODEL", "MAX_ENTROPY_LANGUAGE_MODEL", "MAX_GRADIENT_LANGUAGE_MODEL",
              ""]



def w_l2_distance(w1, w2):
    return np.sqrt(np.sum((w1 - w2) ** 2, axis=1)).mean()


def experiment(stolen_model_fname, original_model_interface, dict_vectorizer, stolen_model,
               minimal_frequency, eta, l2_weight, loss_improvement, maximal_queries_amount, batches_sizes,
               sentences_generator, first_senteneces_generator):
    assert isinstance(stolen_model, sparse_logistic_regression.SparseBinaryLogisticRegression)
    assert isinstance(sentences_generator, inputs_generator.InputGenerator)
    assert ".pkl" not in stolen_model_fname
    print "Loading data"
    print "generating sparse features examples - validation"

    train_sents, dev_sents, test_sents = memm.load_train_dev_test_sentences(DATA_PATH, minimal_frequency)

    print "generating sparse features examples - validation"
    validation_list_of_all_probs, tagged_validation_sents = utils.experiment1_generate_training_examples(
            dev_sents[:400], original_model_interface)

    print "creating input for training"
    validation_probs, validation_sparse_features, validation_predictions = \
        memm.transform_input_for_training(dict_vectorizer, validation_list_of_all_probs,
                                          tagged_validation_sents)
    validation_indices = [[x for x in vec.indices if vec[0, x]] for vec in validation_sparse_features]
    assert validation_probs.shape[0] == len(validation_predictions)
    l2_distances = []
    validation_kl_values = []
    queries_amounts = []
    unique_words_amounts = []
    l2_distance_from_previous_w = []
    accuracies = []

    all_words_queried = set()

    all_training_predictions = []
    all_training_sparse_features = []
    all_training_indices = []

    single_word_queries_amount = 0
    previous_stolen_w = stolen_model.w()
    for batch_size in batches_sizes:
        if single_word_queries_amount >= maximal_queries_amount:
            break
        print 'generating ', batch_size, 'sentences...'
        if single_word_queries_amount == 0:
            print 'first sentences generation'
            new_sentences = [first_senteneces_generator.generate_input() for _ in xrange(batch_size)]
        else:
            new_sentences = [sentences_generator.generate_input() for _ in xrange(batch_size)]
        for s in new_sentences:
            all_words_queried.update(s)
            single_word_queries_amount += len(s)

        new_probs = []
        new_tagged_sentences = []
        for sentence in new_sentences:
            sent_probs, tagged_sentence = original_model_interface.predict_proba(sentence)
            new_probs.append(sent_probs)
            new_tagged_sentences.append(tagged_sentence)
        new_probs, new_sparse_features, new_predictios = memm.transform_input_for_training(
                dict_vectorizer, new_probs, new_tagged_sentences)
        all_training_predictions.append(new_predictios)
        all_training_sparse_features.append(new_sparse_features)
        all_training_indices += [[int(x) for x in vec.indices if vec[0, x]] for vec in new_sparse_features]
        training_losses = []
        start_time = time.time()

        for i in xrange(1, 10000):
            current_eta = eta / i ** 0.3
            stolen_model.epoch_optimize(all_training_indices, np.concatenate(all_training_predictions), 50, current_eta,
                                        l2_weight)
            stolen_model.set_w(utils.minimize_rows_norm(stolen_model.w()))
            if i % 10 == 0:
                print 'eta=', current_eta
                accuracy = np.average(stolen_model.predict(validation_indices) == validation_predictions)
                print 'validation accuracy', accuracy
                train_loss = stolen_model.loss(all_training_indices, np.concatenate(all_training_predictions),
                                               l2_weight)
                print 'training_loss', train_loss
                training_losses.append(train_loss)
                if time.time() - start_time > 3600:
                    print 'Training time exceeded, breaking'
                    break
                if len(training_losses) > 10 and train_loss + loss_improvement > np.average(training_losses[-10:-1]):
                    print 'Training loss stopped improving, breaking'
                    break
        print 'Optimization time: {}seconds'.format(time.time() - start_time)

        validation_kl = np.average(scipy.stats.entropy(stolen_model.predict_proba(validation_indices).T,
                                                       validation_probs.T))
        validation_kl_values.append(validation_kl)

        original_w = original_model_interface.get_w()
        stolen_w = stolen_model.w()
        l2_distance_from_previous_w.append(w_l2_distance(stolen_w, previous_stolen_w))

        previous_stolen_w = stolen_w
        average_l2_distance = w_l2_distance(original_w, stolen_w)
        l2_distances.append(average_l2_distance)

        accuracy = np.average(stolen_model.predict(validation_indices) == validation_predictions)
        accuracies.append(accuracy)

        unique_words_amounts.append(len(all_words_queried))
        queries_amounts.append(single_word_queries_amount)

        print 'current training losses'
        print training_losses
        # stolen_model.save(os.path.join(DATA_PATH, "{}_queries{}.pkl".format(stolen_model_fname,
        #                                                                     single_word_queries_amount)))

    print 'unique_words_amounts'
    print unique_words_amounts
    print 'Single word queries amounts'
    print queries_amounts
    print 'accuracies'
    print accuracies
    print 'l2 distances'
    print l2_distances
    print 'validation KL'
    print validation_kl_values
    print 'W shape:'
    print original_w.shape


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

    try:
        args = parser.parse_args(sys.argv[1:])
    except:
        parser.print_help()
        raise
    assert os.path.sep not in args.original_model_file_name
    assert os.path.sep not in args.stolen_model_file_name

    print ' '.join(sys.argv)
    print args

    words = memm.top_words(DATA_PATH, args.minimal_frequency, args.num_words)
    assert len(set(words)) == len(words) == args.num_words
    dict_vectorizer = memm.get_dict_vectorizer(DATA_PATH, None, args.minimal_frequency)
    original_model = sparse_logistic_regression.SparseBinaryLogisticRegression.load(
            os.path.join(DATA_PATH, args.original_model_file_name))
    original_interface = model_interface.ModelInterface(original_model, dict_vectorizer)

    LENGTH = 25
    length_generator = inputs_generator.constant_generator(LENGTH)
    batches_sizes = [args.first_random / LENGTH] + [args.batch_size / LENGTH] * (
    (args.total_queries_amount - args.first_random) // args.batch_size)

    sentences_generator = None
    shape = original_model.w().shape
    print 'shape', shape

    stolen_model = sparse_logistic_regression.SparseBinaryLogisticRegression(10, shape[0], shape[1])
    iid_generator = inputs_generator.GreedyInputsGenerator(length_generator,
                                                           randomizers.RandomizeByFrequenciesIIDFromDict(
                                                                   {w: 1 for w in words}),
                                                           inputs_generator.TrivialInputScorer(), 1)
    first_senteneces_generator = iid_generator
    train_freq = memm.get_train_count(DATA_PATH)
    train_freq_clipped = {w: train_freq[w] for w in words}
    proportional_words_randomizer = randomizers.RandomizeByFrequencyProportionaly(
            train_freq_clipped, 1.05)
    if args.strategy == "UNIGRAMS":
        scorer = inputs_generator.TrivialInputScorer()
        sentences_generator = inputs_generator.GreedyInputsGenerator(length_generator,
                                                                     proportional_words_randomizer,
                                                                     scorer, 1)
    if args.strategy == "MAX_SIGNIFICANCE":
        scorer = inputs_generator.SubtleDecision(dict_vectorizer, stolen_model, original_interface)
        sentences_generator = inputs_generator.GreedyInputsGenerator(length_generator,
                                                                     proportional_words_randomizer,
                                                                     scorer, 10)
    elif args.strategy == "MAX_GRADIENT":
        scorer = inputs_generator.MaximalGradient(dict_vectorizer, stolen_model, original_interface)
        sentences_generator = inputs_generator.GreedyInputsGenerator(length_generator,
                                                                     proportional_words_randomizer,
                                                                     scorer, 10)
    elif args.strategy == "MAX_ENTROPY":
        scorer = inputs_generator.MaxEntropy(dict_vectorizer, stolen_model, original_interface)
        sentences_generator = inputs_generator.GreedyInputsGenerator(length_generator,
                                                                     proportional_words_randomizer,
                                                                     scorer, 10)
    elif args.strategy == "FROM_TRAIN_SET":
        sentences = map(data.untag, memm.preprocessed_train_use_words(DATA_PATH, words))
        sentences_generator = inputs_generator.SubsetInputsGenerator(sentences)
    elif args.strategy == "IID_WORDS":
        sentences_generator = iid_generator
    elif args.strategy == "SEQUENTIAL":
        sentences_generator = inputs_generator.SequentialInputsGenerator(length_generator, words)

    experiment(args.stolen_model_file_name, original_interface,
               dict_vectorizer, stolen_model, args.minimal_frequency,
               args.eta, args.l2_weight, args.loss_improvement, args.total_queries_amount,
               batches_sizes, sentences_generator, first_senteneces_generator)


if __name__ == '__main__':
    main()
