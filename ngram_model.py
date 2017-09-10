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

        numbers_sentences = self._map_sentences(sentences)
        self._trigram_counts, self._bigram_counts, self._unigram_counts, self._token_count = self._train(numbers_sentences)
        self._train_sentences_count = sum([count for bigram, count in self._bigram_counts.iteritems()
                                     if bigram[0] == self._begin_token])
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

    def _trigram_prob(self, trigram):
        prob = 0
        bigram = trigram[-2:]
        unigram = trigram[-1:]
        prob += self._lambda3() * self._unigram_counts[unigram] / float(self._token_count)
        # If first word in sentence
        if trigram[:2] == (self._begin_token, self._begin_token):
            prob += self._lambda2 * self._bigram_counts.get(bigram, 0) / float(self._train_sentences_count)
            prob += self._lambda1 * self._trigram_counts.get(trigram, 0) / float(self._train_sentences_count)
        else:
            if bigram in self._bigram_counts:
                prob += self._lambda2 * self._bigram_counts[bigram] / float(self._unigram_counts[bigram[:-1]])
            if trigram in self._trigram_counts:
                prob += self._lambda1 * self._trigram_counts[trigram] / float(self._bigram_counts[trigram[:-1]])
        return prob

    def _map_sentences(self, sentences):
        return [[self._begin_token] * 2 + [self._word_to_num[w] for w in s]
                + [self._stop_token] for s in sentences]

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
        print 'time counting', time.time() - start_unigram
        ### END YOUR CODE
        assert token_count == sum(unigram_counts.itervalues())
        assert token_count == sum(bigram_counts.itervalues())
        assert token_count == sum(trigram_counts.itervalues())

        return trigram_counts, bigram_counts, unigram_counts, token_count

    def set_lambda(self, lambda1, lambda2):
        assert lambda1 + lambda2 <= 1.
        self._lambda1 = lambda1
        self._lambda2 = lambda2

    def evaluate_ngrams(self, dataset):
        """
        Goes over an evaluation dataset and computes the perplexity for it with
        the current counts and a linear interpolation
        """
        eval_dataset = self._map_sentences(dataset)
        ### YOUR CODE HERE
        lambda3 = 1 - self._lambda1 - self._lambda2
        # Enforce self._lambda3 > 0, to avoid sentences with probability 0 due to rare bi-grams.
        assert self._lambda1 >= 0 and self._lambda2 >= 0 and lambda3 > 0
        eval_token_count = 0

        assert self._train_sentences_count == sum([count for trigram, count in self._trigram_counts.iteritems()
                                             if trigram[:2] == (self._begin_token, self._begin_token)])
        sum_log = 0
        for sentence in eval_dataset:
            assert self._begin_token == sentence[0] and self._begin_token == sentence[1] and self._stop_token == sentence[-1]
            # Consider all the actual words, and the STOP token.
            eval_token_count += len(sentence) - 2

            n = len(sentence)
            for i in xrange(n - 2):
                trigram = tuple(sentence[i: i + 3])
                prob = self._trigram_prob(trigram)
                sum_log += np.log2(prob)

        perplexity = 2 ** (-1. / eval_token_count * sum_log)
        ### END YOUR CODE
        return perplexity

    def generate_word(self, prefix):
        if len(prefix) < 2:
            prefix = [NGramModel.BEGIN_TOKEN_WORD] * 2 + prefix
        prefix_numbers = tuple(self._word_to_num[w] for w in prefix)
        unigram = prefix_numbers[-1:]
        bigram = prefix_numbers[-2:]

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

# def test_ngram():
#     """
#     Use this space to test your n-gram implementation.
#     """
#     ### YOUR CODE HERE
#     A = (max(self._begin_token, self._stop_token) + 2) % vocabsize
#     B = (A + 1) % vocabsize
#     C = (B + 1) % vocabsize
#     assert len({A, B, C, self._begin_token, self._stop_token}) == 5
#     test_dataset = np.array([[self._begin_token, self._begin_token, A, B, C, self._stop_token],
#                              [self._begin_token, self._begin_token, A, B, self._stop_token],
#                              [self._begin_token, self._begin_token, A, self._stop_token],
#                              [self._begin_token, self._begin_token, self._stop_token]
#                              ])
#     trigram_test, bigram_test, unigram_test, token_count_test = train_ngrams(test_dataset)
#     assert token_count_test == 4 + 3 + 2 + 1
#     print unigram_test
#     assert unigram_test == {(self._stop_token,): 4, (A,): 3, (B,): 2, (C,): 1}
#     assert bigram_test == {(self._begin_token, A): 3, (A, B): 2, (B, C): 1,
#                            (C, self._stop_token):1,
#                            (B, self._stop_token): 1, (A, self._stop_token): 1, (self._begin_token, self._stop_token): 1}
#     assert trigram_test == {(self._begin_token, self._begin_token, A): 3,
#                             (self._begin_token, A, B): 2,
#                             (self._begin_token, A, self._stop_token): 1,
#                             (A, B, C): 1,
#                             (A, B, self._stop_token): 1,
#                             (B, C, self._stop_token): 1,
#                             (self._begin_token, self._begin_token, self._stop_token): 1}

    ### END YOUR CODE



if __name__ == "__main__":
    pass

