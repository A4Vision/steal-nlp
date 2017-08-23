import numpy as np
from hw3 import memm
from hw3 import utils


global_timer = utils.Timer("global_model_interface")


class ModelInterface(object):
    """
    Interface accessible by the attacker.
    Input: A whole sentence.
    Output of predict(): Greedy tagging of the sentence.
    Output of predict_proba(): Tagging probablity vector for each word,
        based on a greedy decoding.
    """

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

    def predict(self, sentence):
        """

        :param sentence:
        :return: tagged sentence
        """
        assert self._sentences_filter(sentence)
        return self.predict_proba(sentence)[1]

    def predict_proba(self, sentence):
        # TODO: Consider using some caching technique here - in case one needs to
        # decode a sentence's prefixes from left to right.
        assert self._sentences_filter(sentence)
        tagged_sentence = [[word, ''] for word in sentence]
        all_probs = []
        for i in xrange(len(sentence)):
            global_timer.start_part("extract_features")
            features = memm.extract_features(tagged_sentence, i)
            global_timer.start_part("vectorize")
            vec_features = memm.vectorize_features(self._dict_vectorizer, features)
            global_timer.start_part("predict_proba")
            probs = self._model.predict_proba(vec_features)
            global_timer.start_part("global")
            all_probs.append(probs[0])
            tag_index = np.argmax(probs)
            tagged_sentence[i][1] = memm.index_to_tag_dict[tag_index]

        return np.array(all_probs), tagged_sentence
