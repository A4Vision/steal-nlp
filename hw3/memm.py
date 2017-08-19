import os
import utils
import sys
import time
import cPickle
import collections
import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from hw3.data import *

BEGIN_TAG = '*'
END_TAG = 'STOP'

tag_to_index_dict, index_to_tag_dict = {}, {}


def get_tag(index):
    return index_to_tag_dict[index]


def get_index(tag):
    return tag_to_index_dict[tag]


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Rerutns: The word's features.
    """
    features = {}
    ### YOUR CODE HERE
    # All words
    features['word'] = curr_word
    features['next_word'] = next_word
    features['prev_word'] = prev_word
    features['prevprev_word'] = prevprev_word

    # All tags
    features['prev_tag'] = prev_tag
    features['prevprev_tag'] = prevprev_tag
    features['tag_bigram'] = prev_tag + ' ' + prevprev_tag

    # Capitalized
    features['is_cap'] = curr_word[0].isupper() and prev_tag != BEGIN_TAG
    features['prev_is_cap'] = prev_word[0].isupper() and prevprev_tag != BEGIN_TAG
    features['next_is_cap'] = next_word[0].isupper()


    # To reduce features amount, we did not use the folowing features:
    # features['prev_word_tag'] = prev_word + ' ' + prev_tag
    # features['prevprev_word_tag'] = prevprev_word + ' ' + prevprev_tag
    if not features['is_cap']:
        curr_word = curr_word.lower()
        for suffix in SUFFIXES:
            features['suffix_{}'.format(suffix)] = int(curr_word.endswith(suffix))

        for prefix in PREFIXES:
            features['prefix_{}'.format(prefix)] = int(curr_word.startswith(prefix))

    ### END YOUR CODE
    return features


def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', BEGIN_TAG)
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', BEGIN_TAG)
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', END_TAG)
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])


def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Rerutns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)


def create_examples(sents):
    global tag_to_index_dict
    print "building examples"
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_index_dict[sent[i][1]])
    return examples, labels

t1_timer = t = utils.Timer("None")

def count_word_tags(tagged_sentences):
    words_counter = collections.Counter()
    tags_counter = collections.Counter()
    for sent in tagged_sentences:
        for w, t in sent:
            words_counter[w] += 1
            tags_counter[t] += 1
    return words_counter, tags_counter


def get_dict_vectorizer(examples):
    DICT_VEC_FNAME = "data/dict_vec.pkl"
    if os.path.exists(DICT_VEC_FNAME):
        vec = cPickle.load(open(DICT_VEC_FNAME, "rb"))
        return vec
    else:
        vec = DictVectorizer()
        vec.fit_transform(examples)
        with open(DICT_VEC_FNAME, "wb") as f_out:
            cPickle.dump(vec, f_out)
        return vec


def load_data(num_dev_sents, num_train_sents):
    global tag_to_index_dict, index_to_tag_dict
    random.seed(123)
    original_train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")[:num_train_sents]
    original_dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")[:num_dev_sents]
    random.shuffle(original_dev_sents)
    random.shuffle(original_train_sents)
    vocab = compute_vocab_count(original_train_sents)
    train_sents = preprocess_sent(vocab, original_train_sents)
    dev_sents = preprocess_sent(vocab, original_dev_sents)

    # The log-linear model training.
    # NOTE: this part of the code is just a suggestion! You can change it as you wish!
    tags = set([token[1] for train_sent in train_sents for token in train_sent])
    # Sort the tags, to make the indexing consistent among different runs.
    tags = sorted(tags)
    tag_to_index_dict = {tag: i for i, tag in enumerate(tags)}
    # Invert the dictionary.
    index_to_tag_dict = {v: k for k, v in tag_to_index_dict.iteritems()}

    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    print "Load vectorizer"
    dict_vectorizer = get_dict_vectorizer(train_examples + dev_examples)
    print "Vectorize examples"
    train_examples_vectorized = dict_vectorizer.transform(train_examples)
    dev_examples_vectorized = dict_vectorizer.transform(dev_examples)

    return dict_vectorizer, count_word_tags(train_sents), train_examples_vectorized, train_labels, dev_examples_vectorized, dev_labels

