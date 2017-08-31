import time
import random
import collections
import numpy as np
import scipy
import scipy.sparse
import theanets
import theano.tensor as TT


class Timer(object):
    def __init__(self, part_name):
        self._times = collections.defaultdict(int)
        self._current = part_name
        self._last = time.time()

    def start_part(self, name):
        self._update()
        self._current = name

    def _update(self):
        self._times[self._current] += time.time() - self._last
        self._last = time.time()

    def __str__(self):
        self._update()
        total = sum(self._times.values())
        res = ''
        for name, elapsed in self._times.iteritems():
            res += '{} {}s {}\n'.format(name, elapsed, elapsed / total)
        return res


def convert_to_sparse((x, y)):
    return scipy.sparse.csr_matrix(x, dtype=np.float64), y


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
        prob = output  # TT.clip(output, eps, 1 - eps)
        actual = self._target
        cross_entropy = -prob * TT.log(actual)
        return cross_entropy.mean()


def predict_from_regression(net, data):
    return np.argmax(net.predict(data), axis=1)


def regression_accuracy(net, data, labels):
    predicted = predict_from_regression(net, data)
    acc = np.sum(predicted == np.array(labels)) / float(len(labels))
    return acc


def regression_kl(net, data, real_probs):
    predicted_probs = net.predict(data)
    return kl(predicted_probs, real_probs)


def kl(probs1, probs2):
    eps = 1e-8
    t = np.clip(probs1, eps, 1 - eps)
    kl = t * np.log(t / np.clip(probs2, eps, 1 - eps))
    return np.abs(kl).mean()


def top_k(l, k):
    a = np.array(l)
    top_k_indices = np.argpartition(a, -k)[-k:]
    return a[top_k_indices]


def minimize_norm(vec):
    return vec - np.average(vec)


def minimize_rows_norm(x):
    for i in xrange(x.shape[0]):
        x[i] = minimize_norm(x[i])
