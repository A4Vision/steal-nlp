import collections
import unittest
from hw3 import inputs_generator
from hw3 import randomizers


class TestRandomizeByFrequencies(unittest.TestCase):
    def test_randomize(self):
        freq = collections.Counter(['a', 'b', 'b'])
        randomizer = randomizers.RandomizeByFrequenciesIIDFromArray(freq)
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
            my_randomizer = randomizers.RandomizeByFrequencyProportionaly(freq, RATIO)
            d = collections.Counter()
            for i in xrange(10):
                x = my_randomizer.random_element()
                my_randomizer.selected_elements([x])
                d[x] += 1

            for i in xrange(20):
                x = my_randomizer.random_element()
                my_randomizer.selected_elements([x])
                d[x] += 1
                ratio = (d[x] - 1) / float(sum(d.values()))
                freq_ratio = freq[x] / float(sum(freq.values()))
                self.assertLessEqual(ratio, freq_ratio * RATIO)

    def test_subset_inputs_generator(self):
        s = ['a', 'b', 'c']
        generated = []
        generator = inputs_generator.SubsetInputsGenerator(s)
        for i in xrange(3):
            generated.append(generator.generate_input())
            if i in (0, 1):
                self.assertEqual(generator.iterations(), 0)
        self.assertListEqual(sorted(generated), s)
        generator.generate_input()
        self.assertEqual(generator.iterations(), 1)
