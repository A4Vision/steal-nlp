import numpy as np
from hw3 import memm


class ModelInterface(object):
    """
    Interface accessible by the attacker.
    Input: A whole sentence.
    Output of predict(): Greedy tagging of the sentence.
    Output of predict_proba(): Tagging probablity vector for each word,
        based on a greedy decoding.
    """

    def __init__(self, model, dict_vectorizer, sentences_filter=lambda: True):
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
        assert self._sentences_filter(sentence)
        return self.predict_proba(sentence)[1]

    def predict_proba(self, sentence):
        # TODO: Consider using some caching technique here - in case one needs to
        # decode a sentence's prefixes from left to right.
        assert self._sentences_filter(sentence)
        tagged_sentence = [[word, ''] for word in sentence]
        all_probs = []
        for i in xrange(len(sentence)):
            features = memm.extract_features(tagged_sentence, i)
            vec_features = memm.vectorize_features(self._dict_vectorizer, features)
            probs = self._model.predict_proba(vec_features)
            all_probs.append(probs)
            tag_index = np.argmax(probs)
            tagged_sentence[i][1] = memm.index_to_tag_dict[tag_index]

        return all_probs, tagged_sentence
