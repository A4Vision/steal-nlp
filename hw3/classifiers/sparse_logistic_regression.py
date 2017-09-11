import numpy as np
import scipy.sparse as S
from hw3 import utils


class SparseBinaryLogisticRegression(object):
    """
    Multinomial logistic regression l2-reguralized model, for sparse binary inputs.
    """

    def __init__(self, data_sparsity, n_features, output_size):
        self._output_size = output_size
        self._n_features = n_features
        self._data_sparsity = data_sparsity

        self._b = np.zeros(output_size, dtype=np.float32)
        self._w = np.float32(np.random.random((n_features, output_size))) * 0.001
        self._w_global_mul = 1.

    def update_w(self):
        print 'updating w !!!'
        self.set_w(self.w())

    def _data_to_arrays(self, data, labels):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.ndim == 2:
            yield data, labels, range(data.shape[0])
        else:
            lens = sorted(set(map(len, data)))
            for l in lens:
                indices = [i for i in xrange(len(data)) if len(data[i]) == l]
                current_data = np.array([data[j] for j in indices])
                # print 'type(current_data)1', type(current_data)
                if labels is None:
                    current_labels = None
                else:
                    current_labels = np.array([labels[j] for j in indices])
                # print 'type(current_data)2', type(current_data)
                yield current_data, current_labels, indices

    def batch_optimize_loss(self, data, labels, eta, reg_coef):
        assert reg_coef >= 0.
        assert data.shape[0] == labels.shape[0]
        if self._w_global_mul < 0.001:
            self.update_w()
        n = data.shape[0]
        # Should average the gradient !
        p = self.predict_proba(data)

        gradient = -p
        gradient[range(gradient.shape[0]), labels] += 1.
        self._b += np.average(gradient, axis=0) * eta

        coef = (1 - 2 * reg_coef * eta)
        gradient *= eta / (self._w_global_mul * coef) / n
        for d, g in zip(data, gradient):
            self._w[d] += g
        self._w_global_mul *= coef

    def epoch_optimize(self, data, labels, batch_size, eta, reg_coef):
        # print 'type(data) 1', type(data)
        for batch_data, batch_labels in utils.iter_batches(batch_size, data, labels):
            self.batch_optimize_loss(batch_data, batch_labels, eta, reg_coef)

    def loss(self, data, labels, reg_coef):
        p = self.predict_proba(data)
        return -np.average(np.log(p[range(p.shape[0]), labels])) + reg_coef * self.w_norm()

    def predict(self, data):
        return np.argmax(self.predict_proba(data), axis=1)

    def predict_proba(self, data):
        res = []
        res_permutation = []
        # print 'type(data) 2', type(data)
        for current_data, _, indices in self._data_to_arrays(data, None):
            # print 'type(current_data) 3', type(current_data)
            # print current_data.flatten()
            res_permutation += indices
            vecs = self._w[current_data.flatten()].reshape(current_data.shape[0], current_data.shape[1], self._w.shape[1])
            before_softmax = np.sum(vecs, axis=1) * self._w_global_mul + self._b
            res.append(softmax_rows(before_softmax))
        # print res[0]
        return np.vstack(res)[utils.invert_permutation(np.array(res_permutation))]

    def set_b(self, b):
        assert b.shape == self._b.shape
        self._b = b.copy()

    def set_w(self, w):
        assert w.shape == self._w.shape
        assert w.dtype == self._w.dtype
        self._w_global_mul = 1.
        self._w = w.copy()

    def w_norm(self):
        return np.sum(self.w() ** 2)

    def w(self):
        return self._w * self._w_global_mul

    def b(self):
        return self._b.copy()

    def save(self, file_or_path):
        if hasattr(file_or_path, 'write'):
            np.save(file_or_path, self.w())
            np.save(file_or_path, self.b())
            np.save(file_or_path, self._data_sparsity)
        else:
            with open(file_or_path, "wb") as f:
                return self.save(f)

    @staticmethod
    def load(file_or_path):
        if hasattr(file_or_path, 'read'):
            w = np.load(file_or_path)
            b = np.load(file_or_path)
            assert w.shape[1] == b.shape[0]
            data_sparsity = np.load(file_or_path)
            res = SparseBinaryLogisticRegression(data_sparsity, w.shape[0], w.shape[1])
            res.set_w(w)
            res.set_b(b)
            return res
        else:
            with open(file_or_path, "rb") as f:
                return SparseBinaryLogisticRegression.load(f)


def softmax_rows(x):
    t = x.T - np.max(x, axis=1)
    t = np.clip(t, -300, np.infty)
    e = np.exp(t)
    return (e / np.sum(e, axis=0)).T


