import collections

import math

import itertools

from data import *

BEGIN_TAG = "<START>"
STOP_TAG = "<STOP>"


def hmm_train(sents):
    """
        sents: list of tagged sentences
        Rerutns: the q-counts and e-counts of the sentences' tags
    """
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = (
        collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int),
        collections.defaultdict(int), collections.defaultdict(int))
    ### YOUR CODE HERE
    # Count the tags
    tags_per_word = collections.defaultdict(set)
    words_count = collections.defaultdict(int)
    for sentence in sents:
        total_tokens += len(sentence) + 1  # count also the STOP.
        for word, tag in sentence:
            e_tag_counts[tag] += 1
            e_word_tag_counts[(word, tag)] += 1
            tags_per_word[word].add(tag)
            words_count[word] += 1
    total_options = 0
    for word, count in words_count.iteritems():
        total_options += count * len(tags_per_word[word])
    print 'average possible tags per word', float(total_options) / sum(words_count.values())

    # Count the triplets, pairs and singles
    for sentence in sents:
        tags = [BEGIN_TAG, BEGIN_TAG] + [tag for word, tag in sentence] + [STOP_TAG]
        # In singles, we count one BEGIN token, and the STOP token.
        for a in tags[1:]:
            q_uni_counts[(a,)] += 1
        # In pairs, we count also the (BEGIN, BEGIN) pair.
        for a, b in zip(tags[:-1], tags[1:]):
            q_bi_counts[(a, b)] += 1
        for a, b, c in zip(tags[:-2], tags[1:-1], tags[2:]):
            q_tri_counts[(a, b, c)] += 1
    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def e(word, tag, e_tag_counts, e_word_tag_counts):
    return e_word_tag_counts[(word, tag)] / float(e_tag_counts[tag])


def q(tag1, tag2, tag3, q_tri_counts, q_bi_counts, q_uni_counts, total_tokens, lambda1, lambda2):
    """
    :param tag1: tags[k-2]
    :param tag2: tags[k-1]
    :param tag3: tags[k]
    :param q_tri_counts:
    :param q_bi_counts:
    :param q_uni_counts:
    :param total_tokens:
    :param lambda1:
    :param lambda2:
    :return:
    """
    lambda3 = 1 - lambda1 - lambda2
    trigram = (tag1, tag2, tag3)
    bigram = (tag2, tag3)
    unigram = (tag1,)
    prob = 0
    prob += lambda3 * q_uni_counts[unigram] / float(total_tokens)
    if bigram in q_bi_counts:
        prob += lambda2 * q_bi_counts[bigram] / float(q_uni_counts[bigram[:-1]])
    if trigram in q_tri_counts:
        prob += lambda1 * q_tri_counts[trigram] / float(q_bi_counts[trigram[:-1]])
    return prob


def hmm_viterbi(sentence_words, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts,
                lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
            considering only the most frequent path, thus optimizing the
            accuracy of the WHOLE sentence.
    """
    lambda3 = 1 - lambda1 - lambda2
    # To avoid probability 0, we require lambda3 > 0
    assert lambda1 >= 0 and lambda2 >= 0 and lambda3 > 0
    all_tags = e_tag_counts.keys()
    if len(sentence_words) == 1:
        return ["JJ"]
    # Literals as in Michael Collins' notes, page 18 on
    #   http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf
    predicted_tags = ["JJ"] * (len(sentence_words))
    ### YOUR CODE HERE
    layer = {(BEGIN_TAG, BEGIN_TAG): 0}
    back_pointers = []
    SMALL = -99999.
    sizes = []
    for word in sentence_words:
        new_layer = collections.defaultdict(lambda: SMALL)
        current_back_pointers = {}
        for v in all_tags:
            if e(word, v, e_tag_counts, e_word_tag_counts) == 0:
                continue
            for (w, u), prev_log_prob in layer.iteritems():
                log_prob_wuv = (prev_log_prob + math.log(q(w, u, v,
                                                           q_tri_counts, q_bi_counts, q_uni_counts, total_tokens,
                                                           lambda1, lambda2))
                                + math.log(e(word, v, e_tag_counts, e_word_tag_counts)))
                if log_prob_wuv > new_layer[(u, v)]:
                    new_layer[(u, v)] = log_prob_wuv
                    current_back_pointers[(u, v)] = w
        back_pointers.append(current_back_pointers)
        layer = {key: value for key, value in new_layer.iteritems() if value > SMALL}
        sizes.append(len(layer))
    two_final_tags = max(layer.items(), key=lambda ((u, v), log_prob):
        log_prob + math.log(q(u, v, STOP_TAG, q_tri_counts, q_bi_counts, q_uni_counts, total_tokens, lambda1, lambda2)))[0]
    predicted_tags = list(two_final_tags)
    for current_back_pointers in back_pointers[::-1][:-2]:
        w = current_back_pointers[tuple(predicted_tags[:2])]
        predicted_tags.insert(0, w)
    ### END YOUR CODE

    return predicted_tags


def hmm_viterbi_marginal(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
            considering marginal probabilities, thus optimizing the
            accuracy of each word separately.
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    return predicted_tags


def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts,
             lambda1, lambda2):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    count_total = 0
    count_good = 0
    for sentence in test_data:
        words = [word for word, tag in sentence]
        hmm_tags = hmm_viterbi(words, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                               e_tag_counts,
                               lambda1, lambda2)
        assert len(hmm_tags) == len(sentence)
        for (word, tag), hmm_tag in zip(sentence, hmm_tags):
            if tag == hmm_tag:
                count_good += 1
            count_total += 1
    acc_viterbi = float(count_good) / count_total
    ### END YOUR CODE
    return acc_viterbi


def main():
    train_sents = read_conll_pos_file("data/Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("data/Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)

    best_lambdas = 0, 0
    best_accuracy = 0
    for l1, l2 in itertools.product([0.97, 0.98, 0.99], [0.005, 0.01, 0.015]):
        if l1 + l2 >= 1:
            continue
        acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,
                               e_tag_counts, l1, l2)
        if acc_viterbi > best_accuracy:
            best_lambdas = (l1, l2)
            best_accuracy = acc_viterbi
    print 'best lambdas', best_lambdas
    print "dev: acc hmm viterbi: {}".format(best_accuracy)

    if os.path.exists("data/Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("data/Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        l1, l2 = best_lambdas
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                               e_word_tag_counts, e_tag_counts, l1, l2)
        print "test: acc hmm viterbi: {}".format(acc_viterbi)

if __name__ == '__main__':
    main()
