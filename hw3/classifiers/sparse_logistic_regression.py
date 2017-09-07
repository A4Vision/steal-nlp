import tempfile
import scipy.sparse as S
import numpy as np
import theanets
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
        for batch_data, batch_labels in utils.iter_batches(batch_size, np.array(data), labels):
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


def check_matrix_efficiency():
    N = 20000
    small, large = 45, 18000
    w_small_rows = np.random.random((large, small))
    x = 123
    d = [1, 100, 10000, 15000, 900, 200, 231, 312]
    timer = utils.Timer("small rows")
    for i in xrange(N):
        vec = np.random.randint(0, large, 10)
        np.sum(w_small_rows[vec], axis=0)
    timer.start_part("global")
    w_small_columns = np.random.random((small, large))
    t = w_small_columns.T.copy()
    np.random.seed(123)
    vec = np.random.randint(0, large, 10)
    print np.sum(t[vec], axis=0)
    c = np.concatenate([vec, vec])
    x = t[c].reshape((2, vec.shape[0], t.shape[1]))
    print np.sum(x, axis=1) - np.sum(t[vec], axis=0)
    timer.start_part("small columns")
    for i in xrange(N):
        vec = np.random.randint(0, large, 10)
        np.sum(t[vec], axis=0)
    timer.start_part("global")
    print timer


def compare_classifier_with_my():
    N = 2 ** 14
    model = theanets.Classifier.load("/home/bugabuga/PycharmProjects/steal-nlp/hw3/data/all_freq20.pkl")
    w = model.params[0].get_value()
    sparisty = 10
    my_model = SparseBinaryLogisticRegression(sparisty, w.shape[0], w.shape[1])
    my_model.set_w(w)
    my_model.set_b(model.params[1].get_value())
    for batch in (2, 10, 20, 50, 100):
        print 'batch', batch
        timer = utils.Timer("global")
        data = np.random.randint(0, w.shape[0], (batch, sparisty))
        data_sparse = S.csr_matrix((batch, w.shape[0]))
        for i, row in enumerate(data):
            data_sparse[i, row] = 1
        timer.start_part("optimized numpy")
        for i in xrange(N):
            p = my_model.predict_proba(data)
        timer.start_part("theanets")
        for i in xrange(N):
            p = model.predict_proba(data_sparse)
        timer.start_part("global")
        print timer


def compare_classifier_with_my_optimization():
    N = 2 ** 14
    model = theanets.Classifier.load("/home/bugabuga/PycharmProjects/steal-nlp/hw3/data/all_freq20.pkl")
    w = model.params[0].get_value()
    print w.shape
    sparisty = 10
    my_model = SparseBinaryLogisticRegression(sparisty, w.shape[0], w.shape[1])
    my_model.set_w(w)
    my_model.set_b(model.params[1].get_value())
    for batch in (10, 20, 50, 100):
        print 'batch', batch
        timer = utils.Timer("global")
        data = np.random.randint(0, w.shape[0], (batch, sparisty))
        labels = np.random.randint(0, w.shape[1], data.shape[0])
        data_sparse = S.csr_matrix((batch, w.shape[0]))
        for i, row in enumerate(data):
            data_sparse[i, row] = 1
        timer.start_part("optimized numpy")
        for i in xrange(N):
            my_model.batch_optimize_loss(data, labels, 0.01, 0.0001)
        timer.start_part("theanets")
        for i in xrange(N):
            p = model.predict_proba(data_sparse)
        timer.start_part("global")
        print timer


def test_optimization():
    N = 2 ** 6
    model = theanets.Classifier.load("/home/bugabuga/PycharmProjects/steal-nlp/hw3/data/all_freq20.pkl")
    model._rng = 13
    w = model.params[0].get_value()
    print w.shape
    sparisty = 10
    my_model = SparseBinaryLogisticRegression(sparisty, w.shape[0], w.shape[1])
    my_model.set_w(w)
    my_model.set_b(model.params[1].get_value())
    for batch in (10, 20, 50, 100):
        print 'batch', batch
        timer = utils.Timer("global")
        data = np.array([np.random.randint(0, w.shape[0], np.random.randint(0, 10)) for i in xrange(batch)])
        labels = np.random.randint(0, w.shape[1], data.shape[0])
        data_sparse = S.csr_matrix((batch, w.shape[0]))
        labels_100 = np.tile(labels, 100)
        data_sparse_100 = S.csr_matrix((batch * 100, w.shape[0]))
        reg = 0.001
        for i, row in enumerate(data):
            data_sparse[i, row] = 1
            for j in xrange(100):
                data_sparse_100[i * 100 + j, row] = 1
        timer.start_part("optimized numpy")
        for i in xrange(N):
            for j in xrange(100):
                my_model.batch_optimize_loss(data, labels, 0.3, reg)
            print my_model.loss(data, labels, reg), my_model.w_norm()
        timer.start_part("theanets")
        count = 0
        for train, valid in model.itertrain([data_sparse_100, np.int32(labels_100)],
                                          algo='sgd',
                                          learning_rate=0.1,
                                          l2_weight=reg):
            count += 1
            if count == N:
                break
        timer.start_part("global")
        print timer


def test_save_load():
    output_size = 10
    input_size = 100
    sparisty = 10
    my_model = SparseBinaryLogisticRegression(sparisty, input_size, output_size)
    my_model.set_w(np.random.random((input_size, output_size)))
    my_model.set_b(np.arange(0, output_size, 0.5)[:output_size])
    f = tempfile.TemporaryFile()
    data = np.random.randint(0, input_size, (50, sparisty))
    labels = np.random.randint(0, output_size, data.shape[0])
    my_model.batch_optimize_loss(data, labels, 0.1, 0.01)
    my_model.batch_optimize_loss(data, labels, 0.1, 0.01)
    my_model.batch_optimize_loss(data, labels, 0.1, 0.001)
    my_model.save(f)
    f.seek(0)
    my_model_loaded = SparseBinaryLogisticRegression.load(f)
    print np.abs(my_model.predict_proba(data) - my_model_loaded.predict_proba(data)).max()


def test_data_to_arrays():
    tmp = SparseBinaryLogisticRegression(10, 100, 10)
    all_indices = []
    res = []
    data = [[1, 2, 3], [4, 5, 6, 7], [1, 4, 6], [1, 455, 2, 23, 23], [111, 3, 4]]
    for current_data, _, indices in tmp._data_to_arrays(data, None):
        all_indices += indices
        res.append([x[:3] for x in current_data])
    t = np.vstack(res)[utils.invert_permutation(np.array(all_indices))]
    print t.tolist()
    assert t.tolist() == [x[:3] for x in data]


if __name__ == '__main__':
    # compare_classifier_with_my_optimization()
    compare_classifier_with_my()
    # test_data_to_arrays()
    # test_optimization()
    # test_save_load()
