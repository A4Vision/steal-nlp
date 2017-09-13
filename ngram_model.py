#!/usr/local/bin/python
import collections
import random
import time
import numpy as np
from hw3 import randomizers

from hw3 import utils


class NGramModel(object):
    BEGIN_TOKEN_WORD = "<s>"
    STOP_TOKEN_WORD = "</s>"

    def __init__(self, sentences, lambda1, lambda2):
        assert lambda1 + lambda2 <= 1.
        self._words = self._all_words(sentences) | {NGramModel.BEGIN_TOKEN_WORD, NGramModel.STOP_TOKEN_WORD}
        self._num_to_word = dict(enumerate(sorted(self._words)))
        self._word_to_num = dict([(v, k) for k, v in self._num_to_word.iteritems()])
        self._begin_token = self._word_to_num[NGramModel.BEGIN_TOKEN_WORD]
        self._stop_token = self._word_to_num[NGramModel.STOP_TOKEN_WORD]

        numbers_sentences = map(self._map_sentence, sentences)
        self._trigram_counts, self._bigram_counts, self._unigram_counts, self._token_count = self._train(numbers_sentences)
        self._train_sentences_count = sum([count for bigram, count in self._bigram_counts.iteritems()
                                     if bigram[0] == self._begin_token])
        assert self._train_sentences_count == sum([count for trigram, count in self._trigram_counts.iteritems()
                                                   if trigram[:2] == (self._begin_token, self._begin_token)])
        self._lambda1 = lambda1
        self._lambda2 = lambda2

        self._unigram_randomizer = self._calc_unigram_randomizer()
        self._bigram_randomizers = self._calc_bigram_randomizers()
        self._trigram_randomizers = self._calc_trigram_randomizers()

    def _calc_unigram_randomizer(self):
        d = {word: count for (word,), count in self._unigram_counts.iteritems()}
        d[self._stop_token] = self._train_sentences_count
        return randomizers.RandomizeByFrequenciesIIDFromDict(d)

    def _calc_bigram_randomizers(self):
        frequencies = {bigram[:1]: {} for bigram in self._bigram_counts}
        for bigram, count in self._bigram_counts.iteritems():
            frequencies[bigram[:-1]][bigram[1]] = count
        return {unigram: randomizers.RandomizeByFrequenciesIIDFromDict(freq_dict)
                for unigram, freq_dict in frequencies.iteritems()}

    def _calc_trigram_randomizers(self):
        frequencies = {trigram[:2]: {} for trigram in self._trigram_counts}
        for trigram, count in self._trigram_counts.iteritems():
            frequencies[trigram[:-1]][trigram[2]] = count
        return {bigram: randomizers.RandomizeByFrequenciesIIDFromDict(freq_dict)
                for bigram, freq_dict in frequencies.iteritems()}

    def _lambda3(self):
        return 1 - self._lambda1 - self._lambda2

    def _map_sentence(self, s):
        return [self._begin_token] * 2 + [self._word_to_num[w] for w in s] + [self._stop_token]

    def _all_words(self, dataset):
        res = set()
        for s in dataset:
            res.update(s)
        return res

    def _train(self, dataset):
        """
            Gets an array of arrays of indexes, each one corresponds to a word.
            Returns trigram, bigram, unigram and total counts.
        """
        trigram_counts = dict()
        bigram_counts = dict()
        unigram_counts = dict()
        token_count = 0
        ### YOUR CODE HERE
        print 'counting ngrams...'
        start_unigram = time.time()
        for sentence in dataset:
            assert self._begin_token == sentence[0] and self._begin_token == sentence[1] and self._stop_token == sentence[-1]
            # Consider all the actual words, and the STOP token.
            token_count += len(sentence) - 2
            n = len(sentence)
            for i in xrange(2, n):
                for length, counter in [(1, unigram_counts), (2, bigram_counts), (3, trigram_counts)]:
                    ngram = tuple(sentence[i - length + 1: i + 1])
                    assert length == len(ngram)
                    counter[ngram] = counter.get(ngram, 0) + 1
        # print 'time counting', time.time() - start_unigram
        ### END YOUR CODE
        assert token_count == sum(unigram_counts.itervalues())
        assert token_count == sum(bigram_counts.itervalues())
        assert token_count == sum(trigram_counts.itervalues())

        return trigram_counts, bigram_counts, unigram_counts, token_count

    def set_lambda(self, lambda1, lambda2):
        assert lambda1 + lambda2 <= 1.
        self._lambda1 = lambda1
        self._lambda2 = lambda2

    def sentences_perplexity(self, sentences):
        """
        Goes over an evaluation dataset and computes the perplexity for it with
        the current counts and a linear interpolation
        """
        sum_log2 = sum(map(self.sentence_log2_likelihood, sentences))
        total_length = sum(map(len, sentences))
        perplexity = 2 ** (-sum_log2 / total_length)
        return perplexity

    def word_prob(self, prefix, word):
        if len(prefix) < 2:
            prefix = [NGramModel.BEGIN_TOKEN_WORD] * 2 + prefix
        prefix_numbers = tuple(self._word_to_num[w] for w in prefix[-2:])
        num = self._word_to_num[word]
        return self._trigram_prob(prefix_numbers + (num,))

    def _trigram_prob(self, numbers_trigram):
        num = numbers_trigram[-1]
        prob = self._lambda3() * self._unigram_randomizer.element_prob(num)

        if numbers_trigram in self._trigram_counts:
            prob += self._lambda1 * self._trigram_randomizers[numbers_trigram[:2]].element_prob(num)
        # Randomize based on bigrams
        if numbers_trigram[1:] in self._bigram_counts:
            prob += self._lambda2 * self._bigram_randomizers[numbers_trigram[1:2]].element_prob(num)
        return prob

    def generate_word(self, prefix):
        if len(prefix) < 2:
            prefix = [NGramModel.BEGIN_TOKEN_WORD] * 2 + prefix
        prefix_numbers = tuple(self._word_to_num[w] for w in prefix[-2:])
        unigram = prefix_numbers[1:]
        bigram = prefix_numbers

        r = random.random()
        # Randomize based on trigrams
        if r < self._lambda1 and bigram in self._trigram_randomizers:
            num = self._trigram_randomizers[bigram].random_element()
        # Randomize based on bigrams
        elif r < self._lambda1 + self._lambda2 and unigram in self._bigram_randomizers:
            num = self._bigram_randomizers[unigram].random_element()
        # Randomize based on unigrams
        else:
            num = self._unigram_randomizer.random_element()
        return self._num_to_word[num]

    def generate_sentence(self, max_length):
        sentence = []
        for i in xrange(max_length):
            word = self.generate_word(sentence)
            if word == NGramModel.STOP_TOKEN_WORD:
                break
            sentence.append(word)
        return [x for x in sentence]

    def sentence_log2_likelihood(self, sentence):
        converted_sentence = self._map_sentence(sentence)
        # Consider all the actual words, and the STOP token.
        triples = np.column_stack((converted_sentence[:-2], converted_sentence[1:-1], converted_sentence[2:]))
        trigrams = map(tuple, triples)
        probs = map(self._trigram_prob, trigrams)
        log2_probs = np.log2(probs)
        return log2_probs.sum()


if __name__ == "__main__":
    pass

