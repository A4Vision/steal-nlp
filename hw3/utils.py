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
    return x


def iter_batches(batch_size, *list_arrays):
    a = list_arrays[0]
    n = a.shape[0]
    for b in list_arrays:
        assert n == b.shape[0]
    permutation = np.random.permutation(n)
    for i in xrange(int(np.ceil(n / float(batch_size)))):
        sub_perm = permutation[i * batch_size: (i + 1) * batch_size]
        yield [b[sub_perm] for b in list_arrays]


def test_iter_batches():
    a = np.random.randint(0, 100, (20, 10))
    b = np.random.randint(0, 100, 20)
    l = list(iter_batches(3, a, b))
    lens = [3, 3, 3, 3, 3, 3, 2]
    assert [(x.shape[0], y.shape[0]) for x, y in l] == zip(lens, lens)
    unitied_a = np.vstack([x for x, y in l])
    unitied_b = np.concatenate([y for x, y in l])
    assert set(unitied_b) == set(b)
    assert set(map(tuple, a)) == set(map(tuple, unitied_a))


def invert_permutation(p):
    '''The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    '''
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


if __name__ == '__main__':
    test_iter_batches()
