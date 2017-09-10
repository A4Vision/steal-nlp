import numpy as np
from hw3 import memm
from hw3 import utils
from hw3.classifiers import sparse_logistic_regression

global_timer = utils.Timer("global_model_interface")
cache_counter = 0
prediction_counter = 0

class ModelInterface(object):
    """
    Interface accessible by the attacker.
    Input: A whole sentence.
    Output of predict(): Greedy tagging of the sentence.
    Output of predict_proba(): Tagging probablity vector for each word,
        based on a greedy decoding.
    """
    END = "_END_"

    def __init__(self, model, dict_vectorizer, sentences_filter=lambda x: True):
        """

        :param model: Predicts the probability for a word to receive a certain tag,
        according to features predefined by memm.extract_features.
        Recieves a sparse vector as input.
        :param dict_vectorizer: DictVectorizer
        :param sentences_filter: Callable that decides whether a sentence is legitimate.
        """
        self._model = model
        self._sentences_filter = sentences_filter
        self._dict_vectorizer = dict_vectorizer

        self._cached_predictions = {}
        self._cached_probs = {}

    def predict(self, sentence):
        """

        :param sentence:
        :return: tagged sentence
        """
        assert self._sentences_filter(sentence)
        if isinstance(self._model, sparse_logistic_regression.SparseBinaryLogisticRegression):
            return self.predict_proba_fast(sentence)[1]
        else:
            return self.predict_proba(sentence)[1]

    def _fill_prev_tags(self, sentence):
        global prediction_counter
        global cache_counter
        long_sentence = sentence + [ModelInterface.END]
        tags = [''] * len(sentence)
        probs = [None] * len(sentence)
        for i in xrange(2, len(sentence) + 2):
            key = tuple(long_sentence[:i])
            if key in self._cached_predictions:
                tags[i - 2] = self._cached_predictions[key]
                probs[i - 2] = self._cached_probs[key]
                cache_counter += 1
            else:
                break
        prediction_counter += len(sentence)
        return tags, probs

    def predict_proba_fast(self, sentence):
        assert self._sentences_filter(sentence)
        long_sentence = sentence + [ModelInterface.END]
        tags, cached_probs = self._fill_prev_tags(sentence)
        # tags, cached_probs = [''] * len(sentence), None
        tagged_sentence = map(list, zip(sentence, tags))

        all_probs = []
        for i in xrange(len(sentence)):
            if tags[i]:
                all_probs.append(cached_probs[i])
                continue
            global_timer.start_part("extract_features")
            features = memm.extract_features(tagged_sentence, i)
            global_timer.start_part("vectorize")
            vec_features = memm.vectorize_features(self._dict_vectorizer, features)
            features_indices = [x for x in vec_features[0].indices if vec_features[0, x]]
            global_timer.start_part("predict_proba")
            probs = self._model.predict_proba([features_indices])[0]
            key = tuple(long_sentence[:i + 2])
            global_timer.start_part("global")
            all_probs.append(probs)
            tag_index = np.argmax(probs)
            tagged_sentence[i][1] = tag = memm.index_to_tag_dict[tag_index]
            self._cached_predictions[key] = tag
            self._cached_probs[key] = probs

        return np.array(all_probs), tagged_sentence

    def predict_proba(self, sentence):
        if isinstance(self._model, sparse_logistic_regression.SparseBinaryLogisticRegression):
            return self.predict_proba_fast(sentence)
        assert self._sentences_filter(sentence)
        tagged_sentence = [[word, ''] for word in sentence]
        all_probs = []
        for i in xrange(len(sentence)):
            global_timer.start_part("extract_features")
            features = memm.extract_features(tagged_sentence, i)
            global_timer.start_part("vectorize")
            vec_features = memm.vectorize_features(self._dict_vectorizer, features)
            global_timer.start_part("predict_proba")
            probs = self._model.predict_proba(vec_features)[0]
            global_timer.start_part("global")
            all_probs.append(probs)
            tag_index = np.argmax(probs)
            tagged_sentence[i][1] = memm.index_to_tag_dict[tag_index]

        return np.array(all_probs), tagged_sentence

    def get_w(self):
        return self._model.w()
