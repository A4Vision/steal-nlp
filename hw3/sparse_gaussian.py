import numpy as np
import scipy.sparse as sparse
import scipy.sparse.csr
from hw3 import utils

class SparseGaussianEliminator(object):
    """
    Maintains a row canonical matrix with sparse rows.
        See https://en.wikipedia.org/wiki/Row_echelon_form#Reduced_row_echelon_form

    Allows to add vectors to the matrix.
    """
    def __init__(self, n, epsilon):
        self._matrix = sparse.csr_matrix([[0] * n], dtype=np.float64)
        self._n = n
        # self._column2row_index[i] = j if the leading column of the row is i.
        self._column2row_index = {}
        self._sparsities = []
        self._eps = epsilon
        self._timer = utils.Timer("global")

    def _is_vector(self, csr_matrix):
        return csr_matrix.shape[0] == 1 and csr_matrix.shape[1] == self._n

    def _leading_column(self, vec):
        assert self._is_vector(vec)
        for i in vec.indices:
            if self._is_nonzero(vec[0, i]):
                return i

    def _is_nonzero(self, x):
        return abs(x) > self._eps

    def _first_eliminatable_column(self, vec):
        for i in vec.indices:
            if self._is_nonzero(vec[0, i]) and i in self._column2row_index:
                return i

    def _eliminate_column(self, vec, j):
        assert isinstance(vec, scipy.sparse.csr.csr_matrix)
        assert self._is_vector(vec)
        assert j in self._column2row_index
        row = self._column2row_index[j]
        assert abs(self._matrix[row, j] - 1.) < 1e-4, (self._matrix[row, j], row, j)
        return vec - self._matrix[row] * (vec[0, j] / self._matrix[row, j])

    def project_vector(self, vec):
        """
        Projects the vector on the space orthogonal to V = Span(matrix).
        :param vec:
        :return:
        """
        assert isinstance(vec, scipy.sparse.csr.csr_matrix)
        assert self._is_vector(vec)
        j = self._first_eliminatable_column(vec)
        while j is not None:
            vec = self._eliminate_column(vec, j)
            j = self._first_eliminatable_column(vec)
        return vec

    def matrix(self):
        return self._matrix[1:].copy()

    def add_vector(self, vec):
        assert isinstance(vec, scipy.sparse.csr.csr_matrix)
        assert self._is_vector(vec)
        projection = self.project_vector(vec)
        j = self._leading_column(projection)
        if j is None:
            return
        new_index = self._matrix.shape[0]
        self._column2row_index[j] = new_index
        column = self._matrix.getcol(j)
        nonzero_in_column = np.diff(column.indptr).nonzero()[0]
        projection /= projection[0, j]
        self._timer.start_part("vstack")
        self._matrix = sparse.vstack((self._matrix, projection))
        self._sparsities.append(projection.indices.shape[0])
        self._timer.start_part("elimination")
        for row in set(nonzero_in_column):
            if self._is_nonzero(self._matrix[row, j]):
                self._matrix[row] -= self._matrix[self._column2row_index[j]] * (vec[0, j] / self._matrix[row, j])
                # self._matrix[row] = self._eliminate_column(self._matrix[row], j)
                self._sparsities[row - 1] = self._matrix[row].indices.shape[0]
            else:
                self._matrix[row, j] = 0.
        self._matrix.eliminate_zeros()
        self._timer.start_part("global")

    def rank(self):
        return self._matrix.shape[0] - 1

    def sparsities(self):
        return self._sparsities

    def timer(self):
        return self._timer
#
# def GEPP(A, B, eps):
#     '''
#     Gaussian elimination with partial pivoting.
#         Solves A * x = B
#     % input: A is an n x n nonsingular matrix
#     '''
#     assert B.shape[0] == A.shape[0]
#     n = A.shape[0]
#
#     def swap_rows(i, j):
#         A[[i, j]] = A[[j, i]]
#         B[[i, j]] = B[[j, i]]
#
#     # k represents the current pivot row. Since GE traverses the matrix in the upper
#     # right triangle, we also use k for indicating the k-th diagonal column index.
#     k = 0
#
#     for column in trange(A.shape[1]):
#         maxindex = abs(A[k:, column]).argmax() + k
#         # print maxindex
#         if abs(A[maxindex, column]) < eps:
#             continue
#         else:
#             pass
#             # print 'found column=', column
#         # Swap rows
#         if maxindex != k:
#             swap_rows(k, maxindex)
#         value = A[k, column]
#         A[k] /= value
#         B[k] /= value
#         for row in xrange(n):
#             if row == k or abs(A[row, column]) < eps:
#                 continue
#             coef = A[row, column]
#             A[row] -= coef * A[k]
#             B[row] -= coef * B[k]
#             # Equation solution column
#         k += 1
#         if k == A.shape[0]:
#             break
#     A[abs(A) < eps] = 0.
#
#     return k, B


