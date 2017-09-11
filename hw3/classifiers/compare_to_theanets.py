import tempfile
import scipy.sparse as S
import scipy
import theanets
import numpy as np
from hw3 import utils
from hw3.classifiers import sparse_logistic_regression


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
    my_model = sparse_logistic_regression.SparseBinaryLogisticRegression(sparisty, w.shape[0], w.shape[1])
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
    my_model = sparse_logistic_regression.SparseBinaryLogisticRegression(sparisty, w.shape[0], w.shape[1])
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
    my_model = sparse_logistic_regression.SparseBinaryLogisticRegression(sparisty, w.shape[0], w.shape[1])
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
    my_model = sparse_logistic_regression.SparseBinaryLogisticRegression(sparisty, input_size, output_size)
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
    my_model_loaded = sparse_logistic_regression.SparseBinaryLogisticRegression.load(f)
    print np.abs(my_model.predict_proba(data) - my_model_loaded.predict_proba(data)).max()


def test_data_to_arrays():
    tmp = sparse_logistic_regression.SparseBinaryLogisticRegression(10, 100, 10)
    all_indices = []
    res = []
    data = [[1, 2, 3], [4, 5, 6, 7], [1, 4, 6], [1, 455, 2, 23, 23], [111, 3, 4]]
    for current_data, _, indices in tmp._data_to_arrays(data, None):
        all_indices += indices
        res.append([x[:3] for x in current_data])
    t = np.vstack(res)[utils.invert_permutation(np.array(all_indices))]
    print t.tolist()
    assert t.tolist() == [x[:3] for x in data]


def measure_my_model():
    input_size = 18000
    output_size = 45
    sparisty = 10
    batch = 50
    model = sparse_logistic_regression.SparseBinaryLogisticRegression(10, input_size, output_size)
    data = np.array([np.random.randint(0, input_size, np.random.randint(sparisty - 3, sparisty)) for i in xrange(10000)])

    x = S.lil_matrix((len(data), input_size), dtype=np.float32)
    for i, r in enumerate(data):
        x[i, r] = 1
    x = x.tocsr()
    model.set_w(np.float32(np.random.random(size=model.w().shape)))
    labels = np.random.randint(0, output_size, size=len(data))
    M = 11
    print M + np.log2(batch * 10 * 45)
    t = utils.Timer("predict_proba")
    for i in xrange(2 ** M):
        l = np.random.randint(0, len(data), batch)
        probs = model.predict_proba(data[l])
    t.start_part("optimize")
    for i in xrange(2 ** M):
        l = np.random.randint(0, len(data), batch)
        model.batch_optimize_loss(data[l], labels[l], 4, 1e-4)
    print t


def check_sparse_multiplication():
    input_size = 18000
    output_size = 45
    sparisty = 10
    w = np.random.random(size=(input_size, output_size))
    data = np.array([np.random.randint(0, input_size, np.random.randint(sparisty - 3, sparisty)) for i in xrange(2 ** 6)])
    x = S.csr_matrix((len(data), input_size))

    for i, r in enumerate(data):
        for j in r:
            x[i, j] = 1
    M = 14
    print np.log2(x.nnz * 45) + M
    t = utils.Timer("sparse mult")
    for i in xrange(2 ** M):
        res = x.dot(w)
    print t
    print res.shape


if __name__ == '__main__':
    # compare_classifier_with_my_optimization()
    # compare_classifier_with_my()
    # test_data_to_arrays()
    # test_optimization()
    # test_save_load()
    measure_my_model()
    # check_sparse_multiplication()
