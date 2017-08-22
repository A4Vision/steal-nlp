import collections
import unittest
from hw3 import inputs_generator


class TestRandomizeByFrequencies(unittest.TestCase):
    def test_randomize(self):
        freq = collections.Counter(['a', 'b', 'b'])
        randomizer = inputs_generator.RandomizeByFrequenciesIID(freq)
        elements = [randomizer.random_element() for i in xrange(2 ** 14)]
        randomizer.selected_elements(elements)  # Practically not required, here for completeness
        c = collections.Counter(elements)
        assert sorted(c.keys()) == ['a', 'b']
        total = sum(c.values())
        self.assertAlmostEqual(c['b'] / float(total), 0.6666, places=2)
        self.assertAlmostEqual(c['a'] / float(total), 0.3333, places=2)

    def test_proportional_randomize(self):
        freq = collections.Counter(['a', 'b', 'b'])
        RATIO = 1.1
        for j in xrange(20):
            randomizer = inputs_generator.RandomizeByFrequencyProportionaly(freq, RATIO)
            d = collections.Counter()
            for i in xrange(10):
                x = randomizer.random_element()
                randomizer.selected_elements([x])
                d[x] += 1

            for i in xrange(20):
                x = randomizer.random_element()
                randomizer.selected_elements([x])
                d[x] += 1
                ratio = (d[x] - 1) / float(sum(d.values()))
                freq_ratio = freq[x] / float(sum(freq.values()))
                self.assertLessEqual(ratio, freq_ratio * RATIO)


