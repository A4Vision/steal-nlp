import time
import random
import collections
import numpy as np


class Timer(object):
    def __init__(self, part_name):
        self._times = collections.defaultdict(int)
        self._current = part_name
        self._last = time.time()

    def start_part(self, name):
        self._update()
        self._current = name

    def _update(self):
        self._times[self._current] += time.time() - self._last
        self._last = time.time()

    def __str__(self):
        self._update()
        total = sum(self._times.values())
        res = ''
        for name, elapsed in self._times.iteritems():
            res += '{} {}s {}\n'.format(name, elapsed, elapsed / total)
        return res


class RandomizeByFrequencies(object):
    def __init__(self, frequencies):
        self._frequencies = frequencies
        items = self._frequencies.items()
        self._elements = [element for element, _ in items]
        counts = [count for _, count in items]
        self._counts_comulative = np.cumsum(counts)
        assert len(self._counts_comulative) == len(counts)

    def random_element(self):
        total = self._counts_comulative[-1]
        index = np.searchsorted(self._counts_comulative, total * random.random())
        return self._elements[index]


class SentencesGenerator(object):
    def __init__(self, word_frequencies):
        self._word_generator = RandomizeByFrequencies(word_frequencies)

    def random_sentence(self, length=None):
        if length is None:
            length = random.randint(10, 20)
        return [self._word_generator.random_element()
                for _ in xrange(length)]


class TaggedSentenceGenerator(object):
    def generate_sentence(self):
        raise NotImplementedError


class FrequenciesTaggedSentenceGenerator(TaggedSentenceGenerator):
    def __init__(self, words_count, tags_count):
        self._words_generator = SentencesGenerator(words_count)
        self._tags_generator = SentencesGenerator(tags_count)

    def generate_sentence(self):
        s = self._words_generator.random_sentence()
        return zip(s, self._tags_generator.random_sentence(len(s)))


