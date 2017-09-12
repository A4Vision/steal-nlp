import scipy.stats
import numpy as np

from hw3 import memm
from hw3 import utils


class TrainingDataAccumulator(object):
    def __init__(self, dict_vectorizer, original_model_interface):
        self._dict_vectorizer = dict_vectorizer
        self._original_model_interface = original_model_interface
        self._all_indices = None
        self._all_predictions = None

    def get_indices_and_labels(self, sentences):
        """
        Returns list of indices and list of labels -
            ready for training.
        :param sentences:
        :return:
        """
        predictions = []
        tagged_sentences = []
        for sentence in sentences:
            sent_probs, tagged_sentence = self._original_model_interface.predict_proba(sentence)
            predictions += np.argmax(sent_probs, axis=1).tolist()
            tagged_sentences.append(tagged_sentence)
        examples, labels = memm.create_examples(tagged_sentences)
        sparse_features = self._dict_vectorizer.transform(examples)
        sparse_features.eliminate_zeros()
        return utils.csr_indices(sparse_features), np.array(labels, dtype=np.int32)

    def query(self, sentences):
        indices, labels = self.get_indices_and_labels(sentences)
        self.add_queries_results(indices, labels)

    def add_queries_results(self, indices, labels):
        assert (self._all_indices is None) == (self._all_predictions is None)
        if self._all_indices is None:
            self._all_indices = indices
            self._all_predictions = labels
        else:
            self._all_indices = np.hstack((self._all_indices, indices))
            self._all_predictions = np.concatenate((self._all_predictions, labels))

    def indices(self):
        return self._all_indices

    def labels(self):
        return self._all_predictions


class StatisticsLogger(object):
    def __init__(self, validation_indices, validations_probs, validation_labels):
        self._validation_indices = validation_indices
        self._validation_probs = validations_probs
        self._validation_labels = validation_labels

        self._validation_kl_values = []
        self._l2_distances = []
        self._w_norms = []
        self._w_norm_percent_from_loss = []
        self._accuracies = []
        self._unique_words_amounts = []
        self._queries_amounts = []

    def log_state(self, stolen_model, original_model_interface, l2_weight, train_loss,
                  n_queries, n_unique_words):
        validation_kl = np.average(scipy.stats.entropy(stolen_model.predict_proba(self._validation_indices).T,
                                                       self._validation_probs.T))
        self._validation_kl_values.append(validation_kl)

        original_w = original_model_interface.get_w()
        stolen_w = stolen_model.w()

        average_l2_distance = w_l2_distance(original_w, stolen_w)
        self._l2_distances.append(average_l2_distance)
        self._w_norms.append(l2_norm(stolen_w))

        self._w_norm_percent_from_loss.append(l2_norm(stolen_w) * l2_weight / train_loss)

        # TODO(bugabuga): Here, maybe one could replace words that are not in the top 2000.
        accuracy = np.average(stolen_model.predict(self._validation_indices) == self._validation_labels)
        self._accuracies.append(accuracy)

        self._queries_amounts.append(n_queries)
        self._unique_words_amounts.append(n_unique_words)

    def print_logged_data(self):
        print 'norm percent from loss'
        print self._w_norm_percent_from_loss
        print 'unique_words_amounts'
        print self._unique_words_amounts
        print 'Single word queries amounts'
        print self._queries_amounts
        print 'accuracies'
        print self._accuracies
        print 'l2 distances'
        print self._l2_distances
        print 'w l2 norm'
        print self._w_norms
        print 'validation KL'
        print self._validation_kl_values
        print 'W shape:'


class BatchDataIterator(object):
    def __init__(self, sentences_generator, first_sentences_generator):
        self._sentences_generator = sentences_generator
        self._first_sentences_generator = first_sentences_generator
        self._all_queried_words = set()
        self._nqueries = 0

    def nqueries(self):
        return self._nqueries

    def n_unique_words(self):
        return len(self._all_queried_words)

    def generate_data(self, batches_sizes):
        first = True
        for batch_size in batches_sizes:
            if first:
                new_sentences = self._first_sentences_generator.generate_many_inputs(batch_size)
                first = False
            else:
                self._sentences_generator.clean_cache()
                new_sentences = self._sentences_generator.generate_many_inputs(batch_size)
            new_sentences = [s for s in new_sentences if len(s) > 0]
            for s in new_sentences:
                self._all_queried_words.update(s)
            self._nqueries += sum(map(len, new_sentences))
            yield new_sentences


def l2_norm(x):
    return np.sum(x ** 2)


def w_l2_distance(w1, w2):
    return np.sqrt(np.sum((w1 - w2) ** 2, axis=1)).mean()
