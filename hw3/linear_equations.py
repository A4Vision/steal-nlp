import sys
import time
from collections import Counter
import numpy as np
import theanets
import random
from hw3 import inputs_generator
from hw3 import model_interface
from hw3 import memm
from hw3 import full_information_experiments
import os
import argparse
np.random.seed(123)
random.seed(123)
parser = argparse.ArgumentParser()
parser.add_argument("--num_words", type=int, required=True)
parser.add_argument("--num_queries", type=int, required=True)
args = parser.parse_args(sys.argv[1:])

train, dev, test = memm.load_train_dev_test_sentences("hw3/data", 20)
c = theanets.Classifier.load(os.path.expanduser("~/Downloads/all_freq20.pkl"))

w_original = c.params[0].get_value()
b_original = c.params[1].get_value()

assert w_original.shape[1] == b_original.shape[0]


def minimize_norm(vec):
    return vec - np.average(vec)


def minimize_rows_norm(x):
    for i in xrange(x.shape[0]):
        x[i] = minimize_norm(x[i])


b_new = minimize_norm(b_original)
minimize_rows_norm(w_original)
c.params[0].set_value(w_original)
c.params[1].set_value(b_new)


dict_vectorizer = memm.get_dict_vectorizer("hw3/data/", None, 20)
interface = model_interface.ModelInterface(c, dict_vectorizer)
count = full_information_experiments.count_words(test)
words = sorted(count.keys(), key=lambda w: -count[w])[:args.num_words]
print 'len(words)', len(words)
gen = inputs_generator.SequentialInputsGenerator(inputs_generator.constant_generator(25), words)
print 'gnerating sentences'
dense_sentences = [gen.generate_input() for _ in xrange(args.num_queries / 25)]
probs = []
tagged = []
print 'quering'
for s in dense_sentences:
    p, t = interface.predict_proba(s)
    tagged.append(t)
    probs.append(p)

print 'transform data'
probs, sparse, predictions = full_information_experiments.transform_input_for_training(dict_vectorizer, probs, tagged)


log_probs = np.log(probs)
special_row = 0
probs_diff = log_probs[:, special_row: special_row + 1] - log_probs
# Each row is [log(p[i]) - log(p[0]) for i in [1, ... n-1]]
B = np.hstack((probs_diff[:, :special_row], probs_diff[:, special_row + 1:]))

all_i_sorted = sorted(set(sparse.indices))
print 'len(all_i)', len(all_i_sorted)


columns = [sparse.getcol(i).toarray() for i in all_i_sorted]
dense_matrix = np.hstack([column for column in columns] + [np.ones(shape=(columns[0].shape[0], 1), dtype=np.float64)])


def l2(x):
    return np.sum(x ** 2)


def is_minimal_vec(vec, epsilon):
    return (l2(vec + epsilon) > l2(vec) and
            l2(vec - epsilon) > l2(vec))


def softmax(x):
    e = np.exp(x - np.max(x))
    assert not np.any(np.isinf(e))
    return e / np.sum(e)


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.
    """

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def projection(vec, ortho_normal_base):
    res = np.zeros_like(vec)
    for x in ortho_normal_base:
        res += np.inner(vec, x) * x
    return res


def minimal_solution(solution, null_base):
    return solution - projection(solution, null_base)


def trange(n):
    start = time.time()
    last_print = start
    for i in xrange(n):
        if time.time() - last_print > 5:
            elapsed = time.time() - start
            left = (n - i) * elapsed / i
            percent = i / float(n) * 100
            last_print = time.time()
            print 'elapsed: {:.1f} left: {:.1f} :: {:.2f}%'.format(elapsed, left, percent)
        yield i


def GEPP(A, B, eps):
    '''
    Gaussian elimination with partial pivoting.
        Solves A * x = B
    % input: A is an n x n nonsingular matrix
    '''
    assert B.shape[0] == A.shape[0]
    n = A.shape[0]

    def swap_rows(i, j):
        A[[i, j]] = A[[j, i]]
        B[[i, j]] = B[[j, i]]
    # k represents the current pivot row. Since GE traverses the matrix in the upper
    # right triangle, we also use k for indicating the k-th diagonal column index.
    k = 0
    
    for column in trange(A.shape[1]):
        maxindex = abs(A[k:, column]).argmax() + k
        # print maxindex
        if abs(A[maxindex, column]) < eps:
            continue
        else:
            pass
            # print 'found column=', column
        #Swap rows
        if maxindex != k:
            swap_rows(k, maxindex)
        value = A[k, column]
        A[k] /= value
        B[k] /= value
        for row in xrange(n):
            if row == k or abs(A[row, column]) < eps:
                continue
            coef = A[row, column]
            A[row] -= coef * A[k]
            B[row] -= coef * B[k]
            # Equation solution column
        k += 1
        if k == A.shape[0]:
            break
    A[abs(A) < eps] = 0.

    return k, B


def solve_triangulated_equations(A, b):
    x = np.zeros(A.shape[1])
    for i, row in enumerate(A):
        nz = row.nonzero()[0]
        if len(nz) > 0:
            leading = nz[0]
            assert abs(row[leading] - 1.) < 0.00001
            x[leading] = b[i]
    return x


d = dense_matrix.copy()
print 'equations', d.shape
print 'eliminating equations matrix'

rank, rhs = GEPP(d, B.copy(), 1e-8)
lhs = d[:rank]
print 'rank', rank


def cut_feature(x):
    if '=' in x:
        return x[:x.index('=')]
    else:
        return x
z = np.zeros(len(dict_vectorizer.vocabulary_))
good_i = [d[i].nonzero()[0][0] for i in xrange(rank) if len(d[i].nonzero()[0]) == 1]
print 'nonzero per row count', Counter(Counter(d.nonzero()[0]).values())
z[[all_i_sorted[x] for x in good_i]] = 1.
good_features = dict_vectorizer.inverse_transform([z])[0].keys()
print 'good features count'
print Counter([cut_feature(x) for x in good_features])


null_base = nullspace(lhs).T
zs = []
for i in xrange(rhs.shape[1]):
    some_solution = solve_triangulated_equations(lhs, rhs[:, i: i + 1])
    zs.append(minimal_solution(some_solution, null_base))
zs = np.vstack(zs).T

# Validating z_i are all solutions to the equations: {lhs * x = rhs}
assert abs(np.dot(lhs, zs) - rhs[:rank, :]).max() < 1e-7
# Validating z_i are all solution to the equations: {dense_matrix * x = B}
assert np.average(abs(np.dot(dense_matrix, zs) - B)) < 1e-10
assert np.max(abs(np.dot(dense_matrix, zs) - B)) < 1e-10

n = zs.shape[1] + 1
w_stolen = np.zeros(shape=(n, zs.shape[0]), dtype=np.float64)
z_sum_div_n = np.sum(zs, axis=1) / n

w_stolen[special_row] = z_sum_div_n.copy()

z_index = 0
for i in xrange(n):
    if i != special_row:
        w_stolen[i] = z_sum_div_n - zs.T[z_index]
        z_index += 1


for row in w_stolen.T:
    assert row.shape == (45,)
    assert is_minimal_vec(row, 1e-6)

V = np.vstack([w_stolen[special_row] - w_stolen[i] for i in xrange(w_stolen.shape[0]) if i != special_row])
assert np.max(abs(np.dot(dense_matrix, V.T) - B)) < 1e-10

w_real = c.params[0].get_value()
b = c.params[1].get_value()
w_real_relevant = np.vstack((w_real[all_i_sorted], b))

V_real = np.vstack([w_real_relevant.T[special_row] - w_real_relevant.T[i] for i in xrange(w_stolen.shape[0]) if i != special_row])
assert np.max(abs(np.dot(dense_matrix, V_real.T) - B)) < 1e-10
for row in w_real_relevant:
    assert row.shape == (45,)
    assert is_minimal_vec(row, 1e-7)

nz = (np.max(np.abs(w_stolen.T - w_real_relevant), axis=1) < 1e-8).nonzero()[0]
print 'Correct w columns:'
print nz
print 'W columns that could be deduced from equations:'
print good_i
