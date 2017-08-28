import os
import sys
import unittest
import numpy as np
from scipy import sparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hw3 import sparse_gaussian



class TestSparseElimination(unittest.TestCase):
    def test_sparse_elimination_basic(self):
        b = sparse.csr_matrix(np.array([[1., 2, 3.],
                                        [4, 5, 6.],
                                        [5, 6, 9]]))
        eliminator = sparse_gaussian.SparseGaussianEliminator(3, 3)
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
                               np.array([[1., 0, -5.],
                                         [0, 1, 4.]])
                               )

    def assertNumpyEquals(self, mat1, mat2):
        return self.assertLess(abs(mat1 - mat2).max(), 1e-10)



if __name__ == '__main__':
    unittest.main()