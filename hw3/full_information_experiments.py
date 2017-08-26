import collections
import os
import argparse
import scipy.sparse
import sys
import theanets
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(BASE_DIR, "data")
sys.path.insert(0, ROOT_DIR)

from hw3 import memm
from hw3 import inputs_generator
from hw3 import model
from hw3 import utils
from hw3 import model_interface


def experiment1_generate_training_examples(wrongly_tagged_sentences, original_model):
    list_of_all_probs = []
    tagged_sentences = []

    for i, tagged_sentence in enumerate(wrongly_tagged_sentences):
        if i % 100 == 0:
            print i / float(len(wrongly_tagged_sentences))
        sentence = [word for word, tag in tagged_sentence]
        probs_vecs, model_tagged_sentence = original_model.predict_proba(sentence)
        list_of_all_probs.append(probs_vecs)
        tagged_sentences.append(model_tagged_sentence)
    return list_of_all_probs, tagged_sentences


def random_subset(l1, l2, n):
    assert l1.shape[0] == len(l2)
    assert n <= l1.shape[0]
    permutation = np.random.permutation(l1.shape[0])
    return l1[permutation[:n]], [l2[i] for i in permutation[:n]]


def count_words(tagged_sentences):
    res = collections.Counter()
    for s in tagged_sentences:
        res.update([w for w, t in s])
    return res


def transform_input_for_training(dict_vectorizer, probs_vecs_list, tagged_sentences):
    probs = np.concatenate(probs_vecs_list)
    examples, labels = memm.create_examples(tagged_sentences)
    sparse_features = dict_vectorizer.transform(examples)

    validation_predictions = np.argmax(probs, axis=1)

    return probs, sparse_features, validation_predictions


def experiment_use_training_set_sentences(model_path, stolen_model_fname, minimal_frequency, batches_sizes,
                                          maximal_queries_amount, sentences_generator, optimization_time):
    assert isinstance(sentences_generator, inputs_generator.InputGenerator)
    assert ".pkl" not in stolen_model_fname
    dict_vectorizer = memm.get_dict_vectorizer(model.DATA_PATH, None, minimal_frequency)
    original_model = model_interface.ModelInterface(theanets.Classifier.load(model_path), dict_vectorizer)
    assert isinstance(original_model, model_interface.ModelInterface)
    original_model._rng = 13
    print "Loading data"
    train_sents, dev_sents, test_sents = memm.load_train_dev_test_sentences(model.DATA_PATH, minimal_frequency)

    print "generating sparse features examples - validation"
    validation_list_of_all_probs, tagged_validation_sents = experiment1_generate_training_examples(dev_sents[:200],
                                                                                                   original_model)

    validation_probs, validation_sparse_features, validation_predictions = \
        transform_input_for_training(dict_vectorizer, validation_list_of_all_probs, tagged_validation_sents)

    input_size = validation_sparse_features.shape[1]
    print "input_size", input_size
    l2_distances = []
    validation_kl_values = []
    queries_amounts = []
    unique_words_amounts = []
    iterations = []
    accuracies = []
    output_size = validation_probs.shape[1]

    layers = [theanets.layers.base.Input(size=input_size, sparse='csr'), (output_size, 'softmax')]

    net = theanets.Regressor(layers,
                             # KL Divergence - Empirically, turns out to give better results than cross entropy.
                             loss='kl')

    learning_rate = 6.
    alpha = 1.
    accuracy = 0.
    all_words_queried = set()

    all_probs = []
    all_sparse_features = []

    single_word_queries_amount = 0
    for batch_size in batches_sizes:
        if single_word_queries_amount >= maximal_queries_amount:
            break
        new_sentences = [sentences_generator.generate_input() for _ in xrange(batch_size)]
        for s in new_sentences:
            all_words_queried.update(s)
            single_word_queries_amount += len(s)

        new_probs = []
        new_tagged_sentences = []
        for sentence in new_sentences:
            sent_probs, tagged_sentence = original_model.predict_proba(sentence)
            new_probs.append(sent_probs)
            new_tagged_sentences.append(tagged_sentence)
        new_probs, new_sparse_features, _ = transform_input_for_training(dict_vectorizer, new_probs,
                                                                         new_tagged_sentences)
        all_probs.append(new_probs)
        all_sparse_features.append(new_sparse_features)
        training_losses = []
        start_time = time.time()

        for train, valid in net.itertrain([scipy.sparse.vstack(all_sparse_features, format='csr'),
                                           np.concatenate(all_probs)],
                                          algo='sgd', learning_rate=learning_rate, weight_l2=alpha):
            accuracy = utils.regression_accuracy(net, validation_sparse_features, validation_predictions).item()
            print 'validation accuracy', accuracy
            # TODO: calculate loss here, and some L2 distances.
            validation_kl = utils.regression_kl(net, validation_sparse_features, validation_probs)
            print 'validation kl', validation_kl
            print 'training_loss', train['loss']
            training_losses.append(train['loss'])
            if time.time() - start_time > optimization_time:
                print 'Optimization time elapsed, breaking'
                break

        original_w = original_model.get_w()
        stolen_w = net.layers[1].find('w').get_value()
        average_l2_distance = np.sqrt(np.sum((original_w - stolen_w) ** 2, axis=1)).mean()
        validation_kl_values.append(validation_kl)
        l2_distances.append(average_l2_distance)
        accuracies.append(accuracy)
        unique_words_amounts.append(len(all_words_queried))
        queries_amounts.append(single_word_queries_amount)
        iterations.append(sentences_generator.iterations())
        print 'current training losses'
        print training_losses
        net.save(os.path.join(DATA_PATH, "{}_queries{}.pkl".format(stolen_model_fname, single_word_queries_amount)))

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
    print 'iterations'
    print iterations
    print 'W shape:'
    print original_w.shape


def main():
    parser = argparse.ArgumentParser(description='Train and steal POS model.')
    parser.add_argument("--classifier_file_name", type=str, help="File name for the original classifier.", required=True)
    parser.add_argument("--minimal_frequency", type=int, help="Minimal frequency for a word to be observed not unknown.", required=True)
    parser.add_argument("--experiment_number", choices=[1, 2, 3, 4], type=int, help="What experiment to run", required=True)
    parser.add_argument("--stolen_fname", help="File name prefix for the stolen models.", required=True, type=str)
    parser.add_argument("--maximal_queries", help="Amounts of data for experiment.", required=True, type=int)
    parser.add_argument("--batch_size", help="Amounts of data for experiment.", required=True, type=int)
    parser.add_argument("--first_batch_size", help="Amounts of queries for first batch.", required=True, type=int)
    parser.add_argument("--search_minutes", help="Maximal minutes per search.", required=True, type=int)
    try:
        args = parser.parse_args(sys.argv[1:])
    except:
        parser.print_help()
        raise
    assert os.path.sep not in args.classifier_file_name
    model_path = os.path.join(model.DATA_PATH, args.classifier_file_name)
    assert os.path.exists(model_path)

    # Pre-process arguments
    batches_sizes = [args.first_batch_size] + [args.batch_size] * int(args.maximal_queries)
    train_sents, _, _ = memm.load_train_dev_test_sentences(model.DATA_PATH, args.minimal_frequency)
    words_freq = count_words(train_sents)

    if args.experiment_number == 1:
        untagged_train_sentences = map(memm.untag_sentence, train_sents)
        generator = inputs_generator.SubsetInputsGenerator(untagged_train_sentences)

        experiment_use_training_set_sentences(model_path, args.stolen_fname, args.minimal_frequency, batches_sizes,
                                              args.maximal_queries, generator, args.search_minutes * 60)
    elif args.experiment_number == 2:
        length = inputs_generator.constant_generator(20)
        generator = inputs_generator.SequentialInputsGenerator(length, set(words_freq.keys()))
        experiment_use_training_set_sentences(model_path, args.stolen_fname, args.minimal_frequency, batches_sizes,
                                              args.maximal_queries, generator, args.search_minutes * 60)
    elif args.experiment_number == 3:
        length = inputs_generator.constant_generator(20)
        words_randomizer = inputs_generator.RandomizeByFrequenciesIIDFromDict({w: 1 for w in words_freq})
        generator = inputs_generator.GreedyInputsGenerator(length, words_randomizer,
                                                           inputs_generator.TrivialInputScorer(), 1)
        experiment_use_training_set_sentences(model_path, args.stolen_fname, args.minimal_frequency, batches_sizes,
                                              args.maximal_queries, generator, args.search_minutes * 60)


if __name__ == '__main__':
    main()

# python hw3/full_information_experiments.py --classifier_file_name classifier_train_all_freq20.pkl --minimal_frequency 20 --experiment_number 1 --stolen_fname stolen20_test --maximal_queries 10000 --first_batch_size 200 --batch_size 200
