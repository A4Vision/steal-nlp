import collections
import functools
import random

from data import *


def most_frequent(counts_dict):
    return max(counts_dict.items(), key=lambda (element, count): count)[0]


def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    ### YOUR CODE HERE
    word2tag_count = collections.defaultdict(functools.partial(collections.defaultdict, int))
    for sentence in train_data:
        for word, tag in sentence:
            word2tag_count[word][tag] += 1
    return {word: most_frequent(tag_count) for word, tag_count in word2tag_count.iteritems()}
    ### END YOUR CODE


def most_frequent_eval(sentences, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    ### YOUR CODE HERE
    count_good = 0
    total_count = 0
    for sentence in sentences:
        for (word, tag) in sentence:
            if pred_tags[word] == tag:
                count_good += 1
            total_count += 1
    return float(count_good) / total_count


def most_frequent_debug_categories(original_sentences, sentences, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    ### YOUR CODE HERE
    count_good = 0
    total_count = 0
    for original_sentence, sentence in zip(original_sentences, sentences):
        for (original_word, _), (word, tag) in zip(original_sentence, sentence):
            if word == original_word:
                continue
            if pred_tags[word] == tag:
                count_good += 1
            # else:
            #     print '{}\t{}\t{}\t{}'.format(original_word, word, tag, pred_tags[word])
            total_count += 1
    return float(count_good) / total_count
    ### END YOUR CODE


if __name__ == "__main__":
    assert most_frequent(collections.Counter([1, 2, 1, 1, 2, 3,])) == 1
    a = read_conll_pos_file("data/Penn_Treebank/train.gold.conll")
    b = read_conll_pos_file("data/Penn_Treebank/dev.gold.conll")
    original_dev_sentences = random.sample(a + b, len(b))
    original_train_sents = [x for x in a + b if x not in original_dev_sentences]

    vocab = compute_vocab_count(original_train_sents)

    train_sents = preprocess_sent(vocab, original_train_sents)
    dev_sents = preprocess_sent(vocab, original_dev_sentences)

    model = most_frequent_train(train_sents)
    print "accuracy on crafted category words (dev): {}".format(
        most_frequent_debug_categories(original_dev_sentences, dev_sents, model))
    print "accuracy on crafted category words (train): {}".format(
        most_frequent_debug_categories(original_train_sents, train_sents, model))

    print "dev: most frequent acc: {}".format(most_frequent_eval(dev_sents, model))

    if os.path.exists('data/Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("data/Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: {}".format(most_frequent_eval(test_sents, model))
