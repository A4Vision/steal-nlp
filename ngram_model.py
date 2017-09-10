#!/usr/local/bin/python
import time
import numpy as np
import pandas as pd


begin_time = time.time()
# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                      index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)
print 'load time', time.time() - begin_time

BEGIN_TOKEN = word_to_num["<s>"]
STOP_TOKEN = word_to_num["</s>"]


def train_ngrams(dataset):
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
        assert BEGIN_TOKEN == sentence[0] and BEGIN_TOKEN == sentence[1] and STOP_TOKEN == sentence[-1]
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
    print 'top common'
    for t, count in sorted(trigram_counts.items(), key=lambda x: -x[1])[:10]:
        print ' '.join(num_to_word[x] for x in t), count, t
    assert token_count == sum(unigram_counts.itervalues())
    assert token_count == sum(bigram_counts.itervalues())
    assert token_count == sum(trigram_counts.itervalues())

    return trigram_counts, bigram_counts, unigram_counts, token_count


def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    ### YOUR CODE HERE
    lambda3 = 1 - lambda1 - lambda2
    # Enforce lambda3 > 0, to avoid sentences with probability 0 due to rare bi-grams.
    assert lambda1 >= 0 and lambda2 >= 0 and lambda3 > 0
    eval_token_count = 0
    train_sentences_count = sum([count for bigram, count in bigram_counts.iteritems()
                                 if bigram[0] == BEGIN_TOKEN])
    assert train_sentences_count == sum([count for trigram, count in trigram_counts.iteritems()
                                         if trigram[:2] == (BEGIN_TOKEN, BEGIN_TOKEN)])
    sum_log = 0
    for sentence in eval_dataset:
        assert BEGIN_TOKEN == sentence[0] and BEGIN_TOKEN == sentence[1] and STOP_TOKEN == sentence[-1]
        # Consider all the actual words, and the STOP token.
        eval_token_count += len(sentence) - 2

        n = len(sentence)
        for i in xrange(2, n):
            trigram = tuple(sentence[i - 2: i + 1])
            bigram = trigram[-2:]
            unigram = trigram[-1:]
            prob = 0
            prob += lambda3 * unigram_counts[unigram] / float(train_token_count)
            if i == 2:
                prob += lambda2 * bigram_counts.get(bigram, 0) / float(train_sentences_count)
                prob += lambda1 * trigram_counts.get(trigram, 0) / float(train_sentences_count)
            else:
                if bigram in bigram_counts:
                    prob += lambda2 * bigram_counts[bigram] / float(unigram_counts[bigram[:-1]])
                if trigram in trigram_counts:
                    prob += lambda1 * trigram_counts[trigram] / float(bigram_counts[trigram[:-1]])
            sum_log += np.log2(prob)

    perplexity = 2 ** (-1. / eval_token_count * sum_log)
    ### END YOUR CODE
    return perplexity


def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    ### YOUR CODE HERE
    A = (max(BEGIN_TOKEN, STOP_TOKEN) + 2) % vocabsize
    B = (A + 1) % vocabsize
    C = (B + 1) % vocabsize
    assert len({A, B, C, BEGIN_TOKEN, STOP_TOKEN}) == 5
    test_dataset = np.array([[BEGIN_TOKEN, BEGIN_TOKEN, A, B, C, STOP_TOKEN],
                             [BEGIN_TOKEN, BEGIN_TOKEN, A, B, STOP_TOKEN],
                             [BEGIN_TOKEN, BEGIN_TOKEN, A, STOP_TOKEN],
                             [BEGIN_TOKEN, BEGIN_TOKEN, STOP_TOKEN]
                             ])
    trigram_test, bigram_test, unigram_test, token_count_test = train_ngrams(test_dataset)
    assert token_count_test == 4 + 3 + 2 + 1
    print unigram_test
    assert unigram_test == {(STOP_TOKEN,): 4, (A,): 3, (B,): 2, (C,): 1}
    assert bigram_test == {(BEGIN_TOKEN, A): 3, (A, B): 2, (B, C): 1,
                           (C, STOP_TOKEN):1,
                           (B, STOP_TOKEN): 1, (A, STOP_TOKEN): 1, (BEGIN_TOKEN, STOP_TOKEN): 1}
    assert trigram_test == {(BEGIN_TOKEN, BEGIN_TOKEN, A): 3,
                            (BEGIN_TOKEN, A, B): 2,
                            (BEGIN_TOKEN, A, STOP_TOKEN): 1,
                            (A, B, C): 1,
                            (A, B, STOP_TOKEN): 1,
                            (B, C, STOP_TOKEN): 1,
                            (BEGIN_TOKEN, BEGIN_TOKEN, STOP_TOKEN): 1}

    ### END YOUR CODE


def argmin(function, parameters_list):
    best_value = 99999999
    best_parameters = None
    for parameters in parameters_list:
        value = function(*parameters)
        if value < best_value:
            best_value = value
            best_parameters = parameters
    return best_parameters, best_value


def grid_search_lambda():
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)

    def perplexity(lambda1, lambda2):
        return evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, lambda1, lambda2)

    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    print "#perplexity: ", perplexity(0.333, 0.533)
    print "unigram perplexity:", perplexity(0, 0)

    start_search = time.time()
    best_no_trigram_perplexity = 99999
    best_lambda2_no_trigram = None
    best_perplexity = 99999
    best_lambda1, best_lambda2 = None, None

    lambda1_values = np.arange(0, 1, 1. / 30).tolist()
    lambda2_values = np.arange(0, 1, 1. / 30).tolist()
    table = HTML.Table(header_row=[None] + ["%.3f" % x for x in lambda1_values])

    for lambda2 in lambda2_values:
        table.rows.append(["%.3f" % lambda2])
        for lambda1 in lambda1_values:
            current = None
            if lambda1 + lambda2 < 1:
                current = perplexity(lambda1, lambda2)
                if current < best_perplexity:
                    best_perplexity = current
                    best_lambda1, best_lambda2 = lambda1, lambda2
                if current < best_no_trigram_perplexity and lambda1 == 0:
                    best_no_trigram_perplexity = current
                    best_lambda2_no_trigram = lambda2
            table.rows[-1].append(current)

    print table
    with open("ngram_grid_search.html", "wb") as fout:
        fout.write(str(table))

    print 'search time', time.time() - start_search
    print "best unigram+bigram perplexity: ", best_no_trigram_perplexity, "lambda2=", best_lambda2_no_trigram
    print "best unigram+bigram+trigram perplexity: ", best_perplexity, "lambda1=", best_lambda1, "lambda2=", best_lambda2


if __name__ == "__main__":
    test_ngram()
    grid_search_lambda()
