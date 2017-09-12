import ngram_model

from hw3 import memm


class DataPreprocessor(object):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'

    def __init__(self, data_path, minimal_frequency, n_top_words, prepare_language_models=True):
        self._minimal_frequenct = minimal_frequency
        self._n_top_words = n_top_words

        self._words = memm.top_words(data_path, minimal_frequency, n_top_words)
        self._train_sents, self._dev_sents, self._test_sents = self._untag(*memm.load_train_dev_test_sentences(data_path, minimal_frequency))
        self._narrow_train_sents, self._narrow_dev_sents, self._narrow_test_sents = self._untag(*memm.load_train_dev_test_sentences_other_preprocess(data_path, self._words))

        if prepare_language_models:
            self._language_model = ngram_model.NGramModel(self._train_sents, 0.4, 0.55)
            self._narrow_language_model = ngram_model.NGramModel(self._narrow_train_sents, 0.4, 0.55)
        else:
            self._language_model = None
            self._narrow_language_model = None

        self._dict_vectorizer = memm.get_dict_vectorizer(data_path, None, minimal_frequency)

    def _untag(self, *sents_lists):
        return [map(memm.untag_sentence, sents) for sents in sents_lists]

    def original_ngram_language_model(self):
        return self._language_model

    def narrow_ngram_language_model(self):
        return self._narrow_language_model

    def original_preprocessed_data(self, data_type):
        return {DataPreprocessor.TRAIN: self._train_sents,
                DataPreprocessor.DEV: self._dev_sents,
                DataPreprocessor.TEST: self._test_sents}[data_type]

    def narrow_preprocessed_data(self, data_type):
        return {DataPreprocessor.TRAIN: self._narrow_train_sents,
                DataPreprocessor.DEV: self._narrow_dev_sents,
                DataPreprocessor.TEST: self._narrow_test_sents}[data_type]

    def dict_vectorizer(self):
        return self._dict_vectorizer

    def words(self):
        return self._words