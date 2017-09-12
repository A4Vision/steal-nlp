import os
import sys
import unittest
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "hw3", "data")
sys.path.insert(0, ROOT_DIR)
from hw3 import data_loader


class TestDataPreprocessor(unittest.TestCase):
    def test_data_loader_sanity_check(self):
        data = data_loader.DataPreprocessor(DATA_PATH, 20, 2000)
        data.dict_vectorizer().transform([{"word": "The"}])
        train = data.original_preprocessed_data('train')
        train_narrow = data.narrow_preprocessed_data('train')
        dev = data.original_preprocessed_data('dev')
        dev_narrow = data.narrow_preprocessed_data('dev')
        words_in_narrow = set()
        for s in train_narrow:
            words_in_narrow.update(s)
        for w in words_in_narrow:
            self.assertIn(w, data.words())
        self.assertLess(18000, len(data.dict_vectorizer().vocabulary_))
        self.assertGreater(19000, len(data.dict_vectorizer().vocabulary_))

        narrow_model = data.narrow_ngram_language_model()
        model = data.original_ngram_language_model()
        print "Calculating perplexities..."
        perplexity_narrow_by_narrow = narrow_model.sentences_perplexity(dev_narrow)
        perplexity_original_by_original = model.sentences_perplexity(dev)
        perplexity_narrow_by_original = model.sentences_perplexity(dev_narrow)
        print perplexity_narrow_by_narrow
        print perplexity_original_by_original
        print perplexity_narrow_by_original

        # This is impossible: perplexity_original_by_narrow = narrow_model.sentences_perplexity(train)
        perplexity_narrow_by_narrow = narrow_model.sentences_perplexity(train_narrow)
        perplexity_original_by_original = model.sentences_perplexity(train)
        perplexity_narrow_by_original = model.sentences_perplexity(train_narrow)
        print perplexity_narrow_by_narrow
        print perplexity_original_by_original
        print perplexity_narrow_by_original

        self.assertLess(perplexity_narrow_by_narrow, 80)
        self.assertLess(perplexity_original_by_original, 80)
        self.assertLess(perplexity_narrow_by_original, 80)
