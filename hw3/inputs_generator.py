import scipy.stats
import numpy as np
import theanets
from hw3 import model_interface
from hw3 import utils
from hw3 import memm


class InputScorer(object):
    def __init__(self, dict_vectorizer, local_model):
        assert isinstance(local_model, theanets.Regressor)
        self._local_model = local_model
        self._dict_vectorizer = dict_vectorizer

    def score(self, sentence, i):
        raise NotImplementedError


class ScoreByCheating(InputScorer):
    def __init__(self, dict_vectorizer, local_model, real_model):
        super(ScoreByCheating, self).__init__(dict_vectorizer, local_model)
        assert isinstance(real_model, model_interface.ModelInterface)

        self._real_model = real_model

    def _cheat_to_get_prev_tags(self, sentence, i):
        tagged_sentence = self._real_model.predict(sentence[:i + 1])
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
        tagged = [[word, ''] for word in sentence[:i + 1]]
        if i > 0:
            tagged[i - 1][1] = prev_tag
        if i > 1:
            tagged[i - 2][1] = prevprev_tag
        return tagged

    def _probs_vec(self, sentence, i):
        tagged_sentence = self._generate_tagged_prefix(sentence, i)
        features_vec = memm.extract_features(tagged_sentence, i)
        probs_vec = self._local_model.predict(features_vec)
        return probs_vec


# These two scoring strategies optimize only for the information held in the feature
# of the current word. Later, we will optimize also for the information held in the features of
# prev_word, and prevprev_word.
class ScoreEntropyByCheating(ScoreByCheating):
    def score(self, sentence, i):
        """
        Entropy of the prediction vector.
        :param sentence:
        :param i:
        :return:
        """
        probs_vec = self._probs_vec(sentence, i)
        return scipy.stats.entropy(probs_vec)


# Note: here, we ignore the requirement of having an easy decision for the model without the current word.
class ScoreSubtleDecisionByCheating(ScoreByCheating):
    def score(self, sentence, i):
        """
        Minus the distance between two highest probabilities.
        :param sentence:
        :param i:
        :return:
        """
        probs_vec = self._probs_vec(sentence, i)
        a, b = utils.top_k(probs_vec, 2)
        return -abs(a - b)

