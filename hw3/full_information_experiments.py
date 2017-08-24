import collections
import os
import argparse
import theano
import sys
import theanets
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(BASE_DIR, "data")
sys.path.insert(0, ROOT_DIR)

from hw3 import memm
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


def experiment1_use_training_set_sentences(model_path, stolen_model_fname, minimal_frequency, word_queries_amounts):
    assert ".pkl" not in stolen_model_fname
    dict_vectorizer = memm.get_dict_vectorizer(model.DATA_PATH, None, minimal_frequency)
    original_model = model_interface.ModelInterface(theanets.Classifier.load(model_path), dict_vectorizer)
    assert isinstance(original_model, model_interface.ModelInterface)
    original_model._rng = 13
    print "Loading data"
    train_sents, dev_sents, test_sents = memm.load_train_dev_test_sentences(model.DATA_PATH, minimal_frequency)

    words_per_sentence = np.average(map(len, train_sents))
    print "generating sparse features examples - training"
    sentences_amount = int(max(word_queries_amounts) / words_per_sentence + 10)
    print sentences_amount
    train_list_of_all_probs, tagged_train_sents = experiment1_generate_training_examples(train_sents[:sentences_amount],
                                                                                         original_model)
    print "generating sparse features examples - validation"
    validation_list_of_all_probs, tagged_validation_sents = experiment1_generate_training_examples(dev_sents[:200],
                                                                                                   original_model)

    validation_probs, validation_sparse_features, validation_predictions = \
        transform_input_for_training(dict_vectorizer, validation_list_of_all_probs, tagged_validation_sents)

    input_size = validation_sparse_features.shape[1]
    print "input_size", input_size
    accuracies = []
    l2_distances = []
    validation_kl_values = []
    output_size = validation_probs.shape[1]

    for single_word_queries_amount in word_queries_amounts:
        print 'queries amount=', single_word_queries_amount
        sentences_queries_amount = int(single_word_queries_amount / words_per_sentence)
        layers = [theanets.layers.base.Input(size=input_size, sparse='csr'), (output_size, 'softmax')]

        net = theanets.Regressor(layers,
                                 # KL Divergence - Empirically, turns out to give better results than cross entropy.
                                 loss='kl')
        current_train_probs_list, current_train_sents = random_subset(np.array(train_list_of_all_probs),
                                                                      tagged_train_sents,
                                                                      sentences_queries_amount)
        words_trained_amount = len(count_words(current_train_sents))
        print 'words_trained_amount', words_trained_amount

        current_train_probs, current_train_sparse_features, _ = \
            transform_input_for_training(dict_vectorizer, current_train_probs_list, current_train_sents)

        learning_rate = 6.
        alpha = 1.
        accuracy = 0.
        loss_function = theano.function(net.variables, outputs=[net.loss(weight_l2=alpha)])
        training_losses = []
        current_accuracies = []
        i = 0
        for train, valid in net.itertrain([current_train_sparse_features, current_train_probs],
                                           algo='sgd', learning_rate=learning_rate, weight_l2=alpha):
            i += 1
            accuracy = utils.regression_accuracy(net, validation_sparse_features, validation_predictions).item()
            print 'validation accuracy', accuracy
            # TODO: calculate loss here, and some L2 distances.
            validation_kl = utils.regression_kl(net, validation_sparse_features, validation_probs)
            print 'validation kl', validation_kl
            train_loss = loss_function(current_train_sparse_features, current_train_probs)[0].item()
            print 'training_loss', train_loss
            training_losses.append(train_loss)
            current_accuracies.append(accuracy)
            less_recent_training = np.average(training_losses[-40:])
            recent_training = np.average(training_losses[-20:])
            if i > 40 and less_recent_training - 1e-4 < recent_training:
                print 'Loss not improving, breaking'
                break
        original_w = original_model.get_w()
        stolen_w = net.layers[1].find('w').get_value()
        validation_kl_values.append(validation_kl)
        average_l2_distance = np.sqrt(np.sum((original_w - stolen_w) ** 2, axis=1)).mean()
        l2_distances.append(average_l2_distance)
        accuracies.append(accuracy)
        print 'current training losses'
        print training_losses
        net.save(os.path.join(DATA_PATH, "{}_queries{}.pkl".format(stolen_model_fname, single_word_queries_amount)))
    print 'accuracies'
    print accuracies
    print 'l2 distances'
    print l2_distances
    print 'validation KL'
    print validation_kl_values


def main():
    parser = argparse.ArgumentParser(description='Train and steal POS model.')
    parser.add_argument("classifier_file_name", type=str, help="File name for the original classifier.")
    parser.add_argument("minimal_frequency", type=int, help="Minimal frequency for a word to be observed not unknown.")
    parser.add_argument("experiment_number", choices=[1, 2, 3, 4], type=int, help="What experiment to run")
    parser.add_argument("--data_amounts", nargs="+", help="Amounts of data for experiment.", required=True)
    parser.add_argument("--stolen_fname", required=True, help="File name prefix for the stolen models.")
    try:
        args = parser.parse_args(sys.argv[1:])
    except:
        parser.print_help()
        raise
    assert os.path.sep not in args.classifier_file_name
    model_path = os.path.join(model.DATA_PATH, args.classifier_file_name)
    assert os.path.exists(model_path)

    if args.experiment_number == 1:
        experiment1_use_training_set_sentences(model_path, args.stolen_fname, args.minimal_frequency, map(int, args.data_amounts))


if __name__ == '__main__':
    main()
