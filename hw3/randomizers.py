import random
import numpy as np


class Randomizer(object):
    def random_element(self):
        raise NotImplementedError

    def selected_elements(self, elements):
        """
        Must be called after selecting an element.
        :param element:
        :return:
        """
        raise NotImplementedError


class RandomizeByFrequenciesIIDFromArray(Randomizer):
    def __init__(self, frequencies_array):
        assert isinstance(frequencies_array, np.ndarray)
        self._counts_comulative = np.cumsum(frequencies_array)
        self._total = float(self._counts_comulative[-1])

    def random_element(self):
        return np.searchsorted(self._counts_comulative, self._total * random.random())

    def selected_elements(self, elements):
        pass

    def element_prob(self, element):
        if element == 0:
            return self._counts_comulative[0] / self._total
        else:
            return (self._counts_comulative[element] - self._counts_comulative[element - 1]) / self._total


class RandomizeByFrequenciesIIDFromDict(Randomizer):
    def __init__(self, frequencies_dict):
        self._index_to_element = sorted(frequencies_dict.keys())
        self._element_to_index = {e: i for i, e in enumerate(self._index_to_element)}

        frequencies_array = np.array([frequencies_dict[element] for element in self._index_to_element])

        self._randomizer = RandomizeByFrequenciesIIDFromArray(frequencies_array)

    def random_element(self):
        return self._index_to_element[self._randomizer.random_element()]

    def selected_elements(self, elements):
        pass

    def element_prob(self, element):
        return self._randomizer.element_prob(self._element_to_index[element])


class RandomizeByFrequencyProportionaly(Randomizer):
    """
    Keeps the distribution of selected elements similar to
    the given frequency.
    Makes sure one does not select the common elements too many times.
    """
    def __init__(self, frequencies_dict, max_ratio):
        # Should be around 1.
        assert 1. < max_ratio < 2.
        self._total_frequencies = sum(frequencies_dict.itervalues())
        self._index_to_element = sorted(frequencies_dict.keys())
        self._element_to_index = {e: i for i, e in enumerate(self._index_to_element)}

        self._frequencies_array = np.array([frequencies_dict[element] for element in self._index_to_element])
        self._frequency_sum = np.sum(self._frequencies_array)

        self._queried_frequencies_array = np.zeros_like(self._frequencies_array, dtype='int')
        self._queried_total = 0

        self._max_ratio = max_ratio

        self._randomizer = RandomizeByFrequenciesIIDFromArray(self._frequencies_array)

    def random_element(self):
        return self._index_to_element[self._randomizer.random_element()]

    def selected_elements(self, elements):
        indices = [self._element_to_index[element] for element in elements]
        # Neglect the option of same element selected more than once.
        self._queried_frequencies_array[indices] += 1
        self._queried_total += len(elements)

        a = self._queried_frequencies_array
        b = self._frequencies_array
        c = self._frequency_sum
        d = self._queried_total

        residual = (self._max_ratio * d * b - c * a) / np.float32(c - self._max_ratio * b)
        # Ignore negatives and division by zero errors.
        residual[np.isnan(residual)] = 0.
        residual = np.max((residual, np.zeros_like(residual)), axis=0)
        self._randomizer = RandomizeByFrequenciesIIDFromArray(residual)
