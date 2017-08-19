import collections
import unittest
import utils


class TestRandomizeByFrequencies(unittest.TestCase):
    def test_randomize(self):
        freq = collections.Counter(['a', 'b', 'b'])
        randomizer = utils.RandomizeByFrequencies(freq)
        c = collections.Counter([randomizer.random_element() for i in xrange(2 ** 14)])
        self.assertListEqual(sorted(c.keys()), ['a', 'b'])
        total = sum(c.values())
        self.assertAlmostEqual(c['b'] / float(total), 0.6666, places=2)
        self.assertAlmostEqual(c['a'] / float(total), 0.3333, places=2)


