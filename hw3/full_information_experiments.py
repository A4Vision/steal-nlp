import os
import argparse
import random
import theano
import sys
import theanets
import numpy as np
from hw3 import memm
from hw3 import model
from hw3 import utils
from hw3 import model_interface


def experiment1_generate_training_examples(dict_vectorizer, wrongly_tagged_sentences, original_model):

    list_of_all_probs = []
    tagged_train_sents = []

    for i, tagged_sentence in enumerate(wrongly_tagged_sentences):
        if i % 100 == 0:
            print i / float(len(wrongly_tagged_sentences))
        sentence = [word for word, tag in tagged_sentence]
        probs_vecs, model_tagged_sentence = original_model.predict_proba(sentence)
        list_of_all_probs.append(probs_vecs)
        tagged_train_sents.append(model_tagged_sentence)
    train_examples, train_labels = memm.create_examples(tagged_train_sents)
    train_sparse_features = dict_vectorizer.transform(train_examples)

    return train_sparse_features, np.concatenate(list_of_all_probs), np.array(train_labels)


def random_subset(l1, l2, n):
    assert l1.shape[0] == l2.shape[0]
    assert n <= l1.shape[0]
    permutation = np.random.permutation(l1.shape[0])
    return l1[permutation[:n]], l2[permutation[:n]]


def experiment1_use_training_set_sentences(model_path, minimal_frequency, data_amounts):
    dict_vectorizer = memm.get_dict_vectorizer(model.DATA_PATH, None, minimal_frequency)
    original_model = model_interface.ModelInterface(theanets.Classifier.load(model_path), dict_vectorizer)
    assert isinstance(original_model, model_interface.ModelInterface)
    original_model._rng = 13
    print "Loading data"
    train_sents, dev_sents, test_sents = memm.load_train_dev_test_sentences(model.DATA_PATH, 123456, 123456, minimal_frequency)

    print "generating sparse features examples - training"
    train_sparse_features, train_probs_vecs, train_labels = experiment1_generate_training_examples(dict_vectorizer, train_sents[:max(data_amounts) / 10], original_model)
    print "generating sparse features examples - validation"
    validation_sparse_features, validation_probs_vecs, validation_labels = experiment1_generate_training_examples(dict_vectorizer, dev_sents[:200], original_model)
    validation_predictions = np.argmax(validation_probs_vecs, axis=1)

    input_size = train_sparse_features.shape[1]
    print "input_size", input_size
    accuracies = []
    output_size = train_probs_vecs.shape[1]
    for data_amount in data_amounts:
        layers = [theanets.layers.base.Input(size=input_size, sparse='csr'), (output_size, 'softmax')]

        net = theanets.Regressor(layers,
                                 # KL Divergence - Empirically, turns out to give better results than corss entropy.
                                 loss='kl')
        current_train_vecs, current_train_probs = random_subset(train_sparse_features, train_probs_vecs, data_amount)
        eta = 20.
        alpha = 1.
        accuracy = 0.
        loss_function = theano.function(net.variables, outputs=[net.loss(weight_l2=alpha)])
        prev_train_loss = 1000.
        for train, valid in net.itertrain([current_train_vecs, current_train_probs],
                                          algo='sgd', learning_rate=eta, weight_l2=alpha):
            accuracy = utils.regression_accuracy(net, validation_sparse_features, validation_predictions)
            print 'validation accuracy', accuracy
            # TODO: calculate loss here, and some L2 distances.
            validation_kl = utils.regression_kl(net, validation_sparse_features, validation_probs_vecs)
            print 'validation kl', validation_kl
            l2 = np.sum(net.params[0].get_value() ** 2)
            train_loss = loss_function(current_train_vecs, current_train_probs)[0]
            print 'training_loss', train_loss

        accuracies.append(accuracy)
    print accuracies


def main():
    parser = argparse.ArgumentParser(description='Train and steal POS model.')
    parser.add_argument("classifier_file_name", type=str, help="File name for the original classifier.")
    parser.add_argument("minimal_frequency", type=int, help="Minimal frequency for a word to be observed not unknown.")
    parser.add_argument("experiment_number", choices=[1, 2, 3, 4], type=int, help="What experiment to run")
    parser.add_argument("--data_amounts", nargs="+", help="Amounts of data for experiment.")
    try:
        args = parser.parse_args(sys.argv[1:])
    except:
        parser.print_help()
        raise
    assert os.path.sep not in args.classifier_file_name
    model_path = os.path.join(model.DATA_PATH, args.classifier_file_name)
    assert os.path.exists(model_path)

    if args.experiment_number == 1:
        experiment1_use_training_set_sentences(model_path, args.minimal_frequency, map(int, args.data_amounts))


if __name__ == '__main__':
    main()
