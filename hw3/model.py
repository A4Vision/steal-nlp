import time
import os
import cPickle
import gzip
import numpy as np
import scipy.sparse
import sys
import theanets
import theanets.util
import theanets.layers.base
import theano.tensor as TT
from sklearn import linear_model
from hw3 import utils
from hw3 import memm


def convert_int32((x, y)):
    return x, np.array(y, dtype=np.int32)


def convert_to_sparse((x, y)):
    return scipy.sparse.csr_matrix(x, dtype=np.float64), y


def transform_data_to_distributions((x, y)):
    z = np.ones((y.shape[0], y.max() + 1)) / 100.
    z[range(y.shape[0]), y] = 1. - 1 / 100. * (y.max())
    return x, z


def convert_labels_to_one_hot(labels):
    z = np.zeros((len(labels), max(labels) + 1), dtype=np.float64)
    z[range(len(labels)), labels] = 1.
    return z


class RegressionCrossEntropy(theanets.losses.Loss):
    __extra_registration_keys__ = ['RXE']

    def __call__(self, outputs):
        output = outputs[self.output_name]
        # eps = 1e-8
        # prob = TT.clip(output, eps, 1 - eps)
        prob = output
        actual = self._target
        cross_entropy = -actual * TT.log(prob)
        return cross_entropy.mean()


class RegressionCrossEntropyInverted(theanets.losses.Loss):
    __extra_registration_keys__ = ['RXE']

    def __call__(self, outputs):
        output = outputs[self.output_name]
        # eps = 1e-8
        prob = output # TT.clip(output, eps, 1 - eps)
        actual = self._target
        cross_entropy = -prob * TT.log(actual)
        return cross_entropy.mean()



def predict_from_regression(net, data):
    return np.argmax(net.predict(data), axis=1)


def regression_accuracy(net, data, labels):
    predicted = predict_from_regression(net, data)
    acc = np.sum(predicted == np.array(labels)) / float(len(labels))
    return acc


def train_mnist_example():
    print 'loading'
    train_data, validation_data, test_data = map(transform_data_to_distributions,
                                                 map(convert_to_sparse,
                                                 map(convert_int32, cPickle.load(gzip.open("/home/bugabuga/PycharmProjects/Steal-ML/data/mnist.pkl.gz")))))

    # 1. create a model -- here, a regression model.
    net = theanets.Regressor([theanets.layers.base.Input(size=train_data[0].shape[1], sparse='csr'), 100,
                              (10, 'softmax')], loss=RegressionCrossEntropy(2))

    # 2. train the model.
    for train, valid in net.itertrain(train_data, valid=validation_data, algo='nag', hidden_l1=0.1,
                                      learning_rate=1.):
        print('training loss:', train['loss'])
        print('most recent validation loss:', valid['loss'])
        acc = regression_accuracy(net, validation_data[0], np.argmax(validation_data[1], axis=1))
        print 'acc=', acc

    # 3. use the trained model.
    acc = regression_accuracy(net, test_data[0], np.argmax(test_data[1], axis=1))
    print 'acc', acc


def get_model(train_data, train_labels, validation_data, validation_labels, model_fname, output_size, hidden_layer_size):
    minimal_accuracy = 0.95
    # 1. create a model -- here, a regression model.
    print "fname", model_fname
    if os.path.exists(model_fname + "..."):
        print "Loading an existing model..."
        net = theanets.Classifier.load(model_fname)
        net._rng = 13
        valid_acc = np.sum(net.predict(validation_data) == np.int32(validation_labels)) / float(len(validation_labels))
        train_acc = np.sum(net.predict(train_data) == np.int32(train_labels)) / float(len(train_labels))
        print "train_acc, valid_acc", train_acc, valid_acc
        if valid_acc > minimal_accuracy:
            return net
    else:
        input_size = train_data.shape[1]
        print "Creating a new model..."

        if hidden_layer_size > 0:
            hidden_layer = dict(name='hidden1', size=hidden_layer_size, std=1. / hidden_layer_size ** 0.5)
            layers = [theanets.layers.base.Input(size=input_size, sparse='csr'), hidden_layer, dict(size=output_size, diagonal=1)]
        else:
            layers = [theanets.layers.base.Input(size=input_size, sparse='csr'), output_size]
        net = theanets.Classifier(layers, loss='xe')

    # 2. train the model.
    print "Training..."
    alpha = 1.
    for eta in [0.01, 0.05]:
        print '(eta, alpha)', (eta, alpha)
        count = 0
        for train, valid in net.itertrain([train_data, np.int32(train_labels)],
                                          valid=[validation_data, np.int32(validation_labels)],
                                          algo='sgd',
                                          learning_rate=eta,
                                          hidden_l1=alpha):
            net.save(model_fname)
            valid_acc = np.sum(net.predict(validation_data) == np.int32(validation_labels)) / float(len(validation_labels))
            print 'valid acc', valid_acc
            count += 1
            if valid_acc > minimal_accuracy or count == 2:
                break

    return net


def generate_random_sentences(words_count, tags_count, amount):
    generator = utils.FrequenciesTaggedSentenceGenerator(words_count, tags_count)
    sentences = [generator.generate_sentence() for i in xrange(amount)]
    l = []
    for s in sentences:
        for i in xrange(len(s)):
            l.append(memm.extract_features(s, i))
    return l


class SentenceScorer(object):
    def score(self, sentence):
        raise NotImplementedError


class ModelInterface(object):
    """
    Interface accessible by the attacker.
    Input: A whole sentence.
    Output of predict(): Greedy tagging of the sentence.
    Output of predict_proba(): Tagging probablity vector for each word,
        based on a greedy decoding.
    """
    def __init__(self, model, dict_vectorizer, sentences_filter=lambda: True):
        """

        :param model: Predicts the probability for a word to receive a certain tag,
        according to features predefined by memm.extract_features.
        Recieves a sparse vector as input.
        :param dict_vectorizer: DictVectorizer
        :param sentences_filter: Callable that decides whether a sentence is legitimate.
        """
        self._model = model
        self._sentences_filter = sentences_filter
        self._dict_vectorizer = dict_vectorizer

    def predict(self, sentence):
        assert self._sentences_filter(sentence)
        return self.predict_proba(sentence)[1]

    def predict_proba(self, sentence):
        assert self._sentences_filter(sentence)
        ### YOUR CODE HERE
        tagged_sentence = [[word, ''] for word in sentence]
        all_probs = []
        for i in xrange(len(sentence)):
            features = memm.extract_features(tagged_sentence, i)
            vec_features = memm.vectorize_features(self._dict_vectorizer, features)
            probs = self._model.predict_proba(vec_features)
            all_probs.append(probs)
            tag_index = np.argmax(probs)
            tagged_sentence[i][1] = memm.index_to_tag_dict[tag_index]

        ### END YOUR CODE
        return all_probs, tagged_sentence


class SamplingSentencesGenerator(utils.TaggedSentenceGenerator):
    def __init__(self, sentences_scorer, tagged_sentences_generator):
        self._underlying_generator = tagged_sentences_generator
        self._sentences_scorer = sentences_scorer

    def generate_sentence(self):
        sentences = [self._underlying_generator.generate_sentence() for i in xrange(100)]
        scores = map(self._sentences_scorer.score, sentences)
        sorted_sentences = sorted(zip(scores, sentences))
        print 'average score:', np.average(scores)
        print 'top scores:', sorted(scores)[-10:]
        # The random sentence with highest score.
        return sorted_sentences[-1][1]


def steal_by_labels(word_count, tag_count, original_model, dict_vectorizer, output_size, validation_examples, hidden_layer_size):
    random_sentences = generate_random_sentences(word_count, tag_count, 10000)
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
        acc = regression_accuracy(net, sparse_features, predictions)
        print 'training accuracy', acc
        validation_acc = regression_accuracy(net, validation_examples, validation_predictions)
        print 'validation accuracy', validation_acc

    return net


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

global_timer = utils.Timer("global")


def only_load_model(num_dev_sents, num_train_sents, model_fname, hidden_layer_size):
    global global_timer
    global_timer.start_part("LoadingData")
    print "* Loading data."
    dict_vectorizer, (word_count, tag_count), train_examples_vectorized, train_labels, dev_examples_vectorized, dev_labels = memm.load_data(num_dev_sents, num_train_sents)
    print global_timer

    global_timer.start_part("CreatingModel")
    print "* Creating original model."
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
    original_model = get_model(train_examples_vectorized, train_labels, dev_examples_vectorized, dev_labels, model_fname, output_size, hidden_layer_size)
    print global_timer

    return original_model


def examine_stealing(num_dev_sents, num_train_sents, model_fname, hidden_layer_size):
    global global_timer
    global_timer.start_part("LoadingData")
    print "* Loading data."
    dict_vectorizer, (word_count, tag_count), train_examples_vectorized, train_labels, dev_examples_vectorized, dev_labels = memm.load_data(num_dev_sents, num_train_sents)
    print global_timer

    global_timer.start_part("CreatingModel")
    print "* Creating original model."
    # assert max(train_labels) == max(dev_labels)
    output_size = max(max(train_labels), max(dev_labels)) + 1
    original_model = get_model(train_examples_vectorized, train_labels, dev_examples_vectorized, dev_labels, model_fname, output_size, hidden_layer_size)
    # Maybe we loaded a model from file, and that model has more possible labels.
    output_size = max(output_size, original_model.find('out', 'b').get_value().shape[0])
    print global_timer

    global_timer.start_part("StealingModel")
    print "* Stealing model."
    stolen_model = steal_by_labels(word_count, tag_count, original_model, dict_vectorizer, output_size, train_examples_vectorized, hidden_layer_size)
    print global_timer

    global_timer.start_part("ComparingModels")
    print "* Comparing models"
    compare_models(stolen_model, original_model, dev_examples_vectorized)
    print global_timer


if __name__ == '__main__':
    # train_mnist_example()
    only_load_model(123456, 5000, "/home/bugabuga/hw3_models/good_classifier_hidden45.pkl", 45)
    # only_load_model(123456, 123456, "/home/bugabuga/hw3_models/good_classifier.pkl", 0)
    # examine_stealing(123456, 123456, "/home/bugabuga/hw3_models/good_classifier_hidden100.pkl", 100)
    # examine_stealing(123456, 123546, "/home/bugabuga/hw3_models/good_classifier.pkl", 0)
