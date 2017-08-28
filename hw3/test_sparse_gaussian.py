import os
import sys
import unittest
import numpy as np
from scipy import sparse
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
from hw3 import sparse_gaussian


class TestSparseElimination(unittest.TestCase):
    def test_sparse_elimination_full_matrix(self):
        b = sparse.csr_matrix(np.array([[1., 2, 3.],
                                        [4, 5, 6.],
                                        [5, 6, 9]]))
        eliminator = sparse_gaussian.SparseGaussianEliminator(3)
        self.assertEqual(eliminator.rank(), 0)
        eliminator.add_vector(b[0])
        self.assertEqual(eliminator.rank(), 1)
        self.assertNumpyEquals(eliminator.matrix().toarray(),
                         b[:1])
        eliminator.add_vector(b[0])
        self.assertEqual(eliminator.rank(), 1)
        self.assertNumpyEquals(eliminator.matrix().toarray(),
                         b[:1])
        eliminator.add_vector(b[1])
        self.assertEqual(eliminator.rank(), 2)
        self.assertNumpyEquals(eliminator.matrix().toarray(),
                               np.array([[1., 0, -1],
                                         [0, 1, 2]])
                               )
        eliminator.add_vector(b[1])
        self.assertEqual(eliminator.rank(), 2)
        eliminator.add_vector(b[2])
        self.assertEqual(eliminator.rank(), 3)
        self.assertNumpyEquals(eliminator.matrix().toarray(),
                               np.array([[1., 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]])
                               )

    def test_elimination_with_missing_columns(self):
        b = sparse.csr_matrix(np.array([[1., 0, 0, 1],
                                        [4, 0, 1, 5],
                                        [0, 0, 0, 1],
                                        [1, 1, 2, 3]]))
        eliminator = sparse_gaussian.SparseGaussianEliminator(4)
        eliminator.add_vector(b[0])
        self.assertNumpyEquals(eliminator.matrix().toarray(),
                               b[:1])
        eliminator.add_vector(b[1])
        self.assertNumpyEquals(eliminator.matrix().toarray(),
                               np.array([[1., 0, 0, 1],
                                         [0, 0, 1, 1]])
                               )
        eliminator.add_vector(b[2])
        self.assertNumpyEquals(eliminator.matrix().toarray(),
                               np.array([[1., 0, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]])
                               )
        eliminator.add_vector(b[3])
        self.assertNumpyEquals(eliminator.matrix().toarray(),
                               np.array([[1., 0, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1],
                                         [0, 1, 0, 0]])
                               )

    def assertNumpyEquals(self, mat1, mat2):
        return self.assertLess(abs(mat1 - mat2).max(), 1e-10, (mat1, mat2))


if __name__ == '__main__':
    unittest.main()