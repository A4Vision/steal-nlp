import random
import collections
import Queue

import scipy.stats
import numpy as np
from hw3 import model_interface
from hw3 import utils
from hw3 import randomizers
from hw3 import data
from hw3 import memm


class InputScorer(object):
    def __init__(self, dict_vectorizer, local_model):
        self._local_model = local_model
        self._dict_vectorizer = dict_vectorizer

    def score(self, sentence, i):
        raise NotImplementedError

    def inform_queried_with(self, sentence):
        """
        Updates the strategy.
        Must be called after a sentence is queried.
        :param sentence:
        :return:
        """
        raise NotImplementedError


class TrivialInputScorer(InputScorer):
    def __init__(self):
        super(TrivialInputScorer, self).__init__(None, None)

    def score(self, sentence, i):
        return 0

    def inform_queried_with(self, sentence):
        pass


class ScoreByCheating(InputScorer):
    # TODO: change inheritance to comparison
    def __init__(self, dict_vectorizer, local_model, real_model):
        super(ScoreByCheating, self).__init__(dict_vectorizer, local_model)
        assert isinstance(real_model, model_interface.ModelInterface)
        self._real_model = real_model

    def _cheat_to_get_prev_tags(self, sentence, i):
        # Instead of cheating, one could replace the rare features
        # with unknown words, to get a good approximation of the tag.
        tagged_sentence = self._real_model.predict(sentence[:i + 2])
        if i > 0:
            prev_tag = tagged_sentence[i - 1][1]
        else:
            prev_tag = memm.BEGIN_TAG
        if i > 1:
            prevprev_tag = tagged_sentence[i - 2][1]
        else:
            prevprev_tag = memm.BEGIN_TAG
        return prevprev_tag, prev_tag

    def _generate_tagged_prefix(self, sentence, i):
        prevprev_tag, prev_tag = self._cheat_to_get_prev_tags(sentence, i)
        tagged = [[word, ''] for word in sentence[:i + 2]]
        if i > 0:
            tagged[i - 1][1] = prev_tag
        if i > 1:
            tagged[i - 2][1] = prevprev_tag
        return tagged

    def _probs_vec(self, sentence, i):
        tagged_sentence = self._generate_tagged_prefix(sentence, i)
        features_vec = memm.extract_features(tagged_sentence, i)
        v = self._dict_vectorizer.transform(features_vec)
        indices = [x for x in v[0].indices if v[0, x]]
        probs_vec = self._local_model.predict_proba([indices])[0]
        return probs_vec


# These two scoring strategies optimize only for the information held in the feature
# of the current word. Later, we will optimize also for the information held in the features of
# prev_word, and prevprev_word.
class MaxEntropy(ScoreByCheating):
    def score(self, sentence, i):
        """
        Entropy of the prediction vector.
        :param sentence:
        :param i:
        :return:
        """
        probs_vec = self._probs_vec(sentence, i)
        return scipy.stats.entropy(probs_vec)

    def inform_queried_with(self, sentence):
        pass


# Note: here, we ignore the requirement of having an easy decision for the model without the current word.
class SubtleDecision(ScoreByCheating):
    def score(self, sentence, i):
        """
        Minus the distance between two highest probabilities.
        :param sentence:
        :param i:
        :return:
        """
        probs_vec = self._probs_vec(sentence, i)
        # TODO: Consider looking on top 3.
        a, b = utils.top_k(probs_vec, 2)
        return -abs(a - b)

    def inform_queried_with(self, sentence):
        pass


class MaximalGradient(ScoreByCheating):
    def score(self, sentence, i):
        """
        Maximize the gradient l2 norm.
        :param sentence:
        :param i:
        :return:
        """
        probs_vec = self._probs_vec(sentence, i)
        return np.sum(probs_vec ** 2)

    def inform_queried_with(self, sentence):
        pass


class SelectWordsUniformly(InputScorer):
    def __init__(self, dict_vectorizer, local_model):
        super(SelectWordsUniformly, self).__init__(dict_vectorizer, local_model)
        # Rough counting - ignoring the subtality of the last and first words.
        self._selected_counter = collections.Counter()
        self._all_words = {feature[5:] for feature in dict_vectorizer.vocabulary_.iterkeys() if
                           feature.startswith("word=")}

    def score(self, sentence, i):
        word = sentence[i]
        if word not in self._all_words:
            return False
        # Always prefer words that were queried less times.
        return -self._selected_counter[word]

    def inform_queried_with(self, sentence):
        self._selected_counter.update(sentence)


def dict_argmax(d):
    """
    Key with highest corresponding value.
    Args:
        d:
    Returns:

    """
    return max((v, k) for k, v in d.items())[1]


class InputGenerator(object):
    def generate_input(self):
        raise NotImplementedError

    def iterations(self):
        return 0

    def clean_cache(self):
        pass


class GreedyInputsGenerator(InputGenerator):
    """
    Generates sentences such that the words are super-uniformly distributed.
    """
    def __init__(self, length_randomizer, words_randomizer, input_scorer, random_tries_per_word):
        assert isinstance(words_randomizer, randomizers.Randomizer)
        assert isinstance(length_randomizer, randomizers.Randomizer)
        assert isinstance(input_scorer, InputScorer)
        assert random_tries_per_word > 0
        self._length_randomizer = length_randomizer
        self._word_randomizer = words_randomizer
        self._scorer = input_scorer
        self._random_tries_per_word = random_tries_per_word

    def generate_input(self):
        length = self._length_randomizer.random_element()
        sentence = [data.UNKNOWN_WORD] * length
        for i in xrange(length):
            words = [self._word_randomizer.random_element() for _ in xrange(self._random_tries_per_word)]
            scores = {}
            for word in words:
                sentence[i] = word
                scores[word] = self._scorer.score(sentence, i)
            sentence[i] = dict_argmax(scores)
        # Inform scorer and randomizer about our selections.
        self._word_randomizer.selected_elements(sentence)
        self._scorer.inform_queried_with(sentence)
        self._length_randomizer.selected_elements([length])
        return sentence


class SequentialInputsGenerator(InputGenerator):
    def __init__(self, length_generator, all_words):
        self._all_words = sorted(set(all_words))
        self._length_generator = length_generator
        self._index = 0
        self._iterations = 0

    def generate_input(self):
        length = self._length_generator.random_element()
        sentence = self._all_words[self._index: self._index + length]
        self._index += length
        if self._index >= len(self._all_words):
            random.shuffle(self._all_words)
            self._index = 0
            self._iterations += 1
        self._length_generator.selected_elements([length])
        return sentence

    def iterations(self):
        return self._iterations


class SubsetInputsGenerator(InputGenerator):
    def __init__(self, sentences):
        self._sentences = list(sentences)
        assert len(self._sentences) > 0
        random.shuffle(self._sentences)
        self._index = 0
        self._iterations = 0

    def generate_input(self):
        res = self._sentences[self._index]
        self._index += 1
        if self._index == len(self._sentences):
            self._index = 0
            self._iterations += 1
        return res

    def iterations(self):
        return self._iterations


def constant_generator(element):
    return randomizers.RandomizeByFrequenciesIIDFromDict({element: 1})


class SelectFromOtherInputsGenerator(InputGenerator):
    """
    Generates sentences using one generator, scores them using a scorer and select only
    the sentences with highest scores.
    """
    def __init__(self, sub_generator, scorer, n_sentences_per_good):
        self._sub_generator = sub_generator
        self._scorer = scorer
        # Previously examined sentences -> score.
        self._best_sentences = Queue.PriorityQueue()
        self._n_sentences_per_good = n_sentences_per_good
        self._selected = set()

    def clean_cache(self):
        self._best_sentences = Queue.PriorityQueue()

    def generate_input(self):
        for _ in xrange(self._n_sentences_per_good):
            sentence = self._sub_generator.generate_input()
            # Avoid using the same sentence twice.
            while id(sentence) in self._selected:
                sentence = self._sub_generator.generate_input()
            self._selected.add(id(sentence))
            score = sum([self._scorer.score(sentence, i) for i in xrange(len(sentence))]) / len(sentence)
            self._best_sentences.put((-score, sentence))
        minus_score, sentence = self._best_sentences.get()
        return sentence


class NGramModelGenerator(InputGenerator):
    def __init__(self, max_length, ngram_model):
        self._max_length = max_length
        self._ngram_model = ngram_model

    def generate_input(self):
        s = self._ngram_model.generate_sentence(self._max_length)
        while len(s) == 0:
            s = self._ngram_model.generate_sentence(self._max_length)
        return s
