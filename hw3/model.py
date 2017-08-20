import random
import os
import numpy as np
import argparse
import sys

import theanets
import theanets.util
import theanets.layers.base
import time

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(BASE_DIR, "data")
sys.path.insert(0, ROOT_DIR)

from hw3 import utils
from hw3 import memm


global_timer = utils.Timer("global")


def create_or_load_model(model_fname, input_size, output_size, hidden_layer_size):
    if os.path.exists(model_fname):
        print "Loading an existing model..."
        net = theanets.Classifier.load(model_fname)
        net._rng = 13

    else:
        print "Creating a new model..."

        if hidden_layer_size > 0:
            hidden_layer = dict(name='hidden1', size=hidden_layer_size, std=1. / hidden_layer_size ** 0.5)
            layers = [theanets.layers.base.Input(size=input_size, sparse='csr'), hidden_layer,
                      dict(size=output_size, diagonal=1)]
        else:
            layers = [theanets.layers.base.Input(size=input_size, sparse='csr'), output_size]
        net = theanets.Classifier(layers, loss='xe')
    return net


def get_model(train_data, train_labels, validation_data, validation_labels, model_fname, output_size,
              hidden_layer_size):
    minimal_accuracy = 0.966
    input_size = train_data.shape[1]
    print "fname", model_fname
    # 1. create/load a model.
    net = create_or_load_model(model_fname, input_size, output_size, hidden_layer_size)

    valid_acc = np.sum(net.predict(validation_data) == np.int32(validation_labels)) / float(len(validation_labels))
    train_acc = np.sum(net.predict(train_data) == np.int32(train_labels)) / float(len(train_labels))
    print "train_acc, valid_acc", train_acc, valid_acc
    if valid_acc > minimal_accuracy:
        return net

    # 2. train the model if not accurate enough.
    print "Training..."
    alpha = 1.
    best_validation = valid_acc
    # This snippet is useful if one wants to tune parameters.
    # random.seed(time.time())  # Seed to avoid two processes writing the same file
    # tmp_filename = "/tmp/first_{}.pkl".format(random.randint(0, 10000))
    # assert not os.path.exists(tmp_filename)
    # net.save(tmp_filename)
    # net = theanets.Classifier.load(tmp_filename)
    # net._rng = 13
    eta = 1.
    print '(eta, alpha)', (eta, alpha)
    prev_valid_acc = valid_acc
    for train, valid in net.itertrain([train_data, np.int32(train_labels)],
                                      valid=[validation_data, np.int32(validation_labels)],
                                      algo='sgd',
                                      learning_rate=eta,
                                      hidden_l1=alpha):
        valid_acc = np.sum(net.predict(validation_data) == np.int32(validation_labels)) / float(
            len(validation_labels))
        if valid_acc > best_validation:
            best_validation = valid_acc
            print "Improved validation accuracy, saving the model !"
            net.save(model_fname)
        elif prev_valid_acc > valid_acc:
            print "Accuracy not improved, shrinking learning rate."
            eta *= 0.8
            print "eta=", eta
        prev_valid_acc = valid_acc
        print 'validation acc', valid_acc
        if valid_acc > minimal_accuracy:
            break
    return net


def only_load_model(num_dev_sents, num_train_sents, model_fname, hidden_layer_size):
    global global_timer
    global_timer.start_part("LoadingData")
    print "* Loading data."
    dict_vectorizer, (word_count,
                      tag_count), train_examples_vectorized, train_labels, dev_examples_vectorized, dev_labels = memm.load_data(
        DATA_PATH,
        num_dev_sents, num_train_sents)
    print global_timer

    global_timer.start_part("CreatingModel")
    print "* Creating original model."
    # An alternative approach to train the logistic regression.
    # print "Running sk learn ! max_iter=256, lbfgs solver"
    # logreg = linear_model.LogisticRegression(
    #     multi_class='multinomial', max_iter=256, solver='lbfgs', C=100000, verbose=1, n_jobs=8)
    # print "Fitting..."
    # start = time.time()
    # logreg.fit(train_examples_vectorized, train_labels)
    # print 'sklearn LogsiticRegression accuracy', np.sum(logreg.predict(dev_examples_vectorized) == np.array(dev_labels)) / float(len(dev_labels))
    # print time.time() - start
    # assert max(train_labels) == max(dev_labels)
    output_size = max(max(train_labels), max(dev_labels)) + 1
    original_model = get_model(train_examples_vectorized, train_labels, dev_examples_vectorized, dev_labels,
                               model_fname, output_size, hidden_layer_size)
    print global_timer

    return original_model


def steal_by_labels(word_count, tag_count, original_model, dict_vectorizer, output_size, validation_examples,
                    hidden_layer_size):
    random_sentences = 3 # generate_random_sentences(word_count, tag_count, 10000)
    sparse_features = dict_vectorizer.transform(random_sentences)

    probs_vecs = original_model.predict_proba(sparse_features)
    predictions = original_model.predict(sparse_features)

    validation_predictions = original_model.predict(validation_examples)

    input_size = sparse_features.shape[1]
    layers = [theanets.layers.base.Input(size=input_size, sparse='csr'), (output_size, 'softmax')]
    if hidden_layer_size > 0:
        layers.insert(1, hidden_layer_size)
    net = theanets.Regressor(layers,
                             # KL Divergence - Empirically, turns out to give better results than corss entropy.
                             loss='kl')
    print "KL Loss"

    eta = 3.
    alpha = 1.
    for train, valid in net.itertrain([sparse_features, probs_vecs],
                                      algo='sgd', learning_rate=eta, hidden_l1=alpha):
        acc = utils.regression_accuracy(net, sparse_features, predictions)
        print 'training accuracy', acc
        validation_acc = utils.regression_accuracy(net, validation_examples, validation_predictions)
        print 'validation accuracy', validation_acc

    return net


def examine_stealing(num_dev_sents, num_train_sents, model_fname, hidden_layer_size, data_path):
    global global_timer
    global_timer.start_part("LoadingData")
    print "* Loading data."
    dict_vectorizer, (word_count,
                      tag_count), train_examples_vectorized, train_labels, dev_examples_vectorized, dev_labels = memm.load_data(
        data_path, num_dev_sents, num_train_sents)
    print global_timer

    global_timer.start_part("CreatingModel")
    print "* Creating original model."
    # assert max(train_labels) == max(dev_labels)
    output_size = max(max(train_labels), max(dev_labels)) + 1
    original_model = get_model(train_examples_vectorized, train_labels, dev_examples_vectorized, dev_labels,
                               model_fname, output_size, hidden_layer_size)
    # Maybe we loaded a model from file, and that model has more possible labels.
    output_size = max(output_size, original_model.find('out', 'b').get_value().shape[0])
    print global_timer

    global_timer.start_part("StealingModel")
    print "* Stealing model."
    stolen_model = steal_by_labels(word_count, tag_count, original_model, dict_vectorizer, output_size,
                                   train_examples_vectorized, hidden_layer_size)
    print global_timer

    global_timer.start_part("ComparingModels")
    print "* Comparing original and stolen model "
    compare_models(stolen_model, original_model, dev_examples_vectorized)
    print global_timer


def compare_models(stolen_model, original_model, test_examples):
    """
    :param stolen_model:
        Regression network, 96120 -> 45
    :param original_model:
        Classifier network, 96120 -> 45
    :param test_examples:
    :return:
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and steal POS model.')
    parser.add_argument("classifier_file_name", type=str)
    parser.add_argument("--valid_size", default=123456, type=int)
    parser.add_argument("--train_size", default=123456, type=int)
    parser.add_argument("--hidden_size", default=0, type=int)
    parser.add_argument("--action", choices=["train", "steal"])
    try:
        args = parser.parse_args(sys.argv[1:])
    except:
        parser.print_help()
        raise
    assert os.path.sep not in args.classifier_file_name
    if args.action == "train":
        only_load_model(args.valid_size, args.train_size, os.path.join(DATA_PATH, args.classifier_file_name),
                        args.hidden_size)
    elif args.action == "steal":
        examine_stealing(args.valid_size, args.train_size, os.path.join(DATA_PATH, args.classifier_file_name),
                         args.hidden_size, DATA_PATH)
