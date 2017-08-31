import utils
import cPickle
import collections
import random
from sklearn.feature_extraction import DictVectorizer
from hw3.data import *

BEGIN_TAG = '*'
END_TAG = 'STOP'

index_to_tag = ['#', '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
                'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
                'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']
index_to_tag_dict = {i: tag for i, tag in enumerate(index_to_tag)}
tag_to_index_dict = {tag: i for i, tag in enumerate(index_to_tag)}


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
    features['next_is_cap'] = next_word[0].isupper() and next_word != END_TAG#added end_tag check


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


def get_dict_vectorizer(data_path, examples, minimal_frequency):
    DICT_VEC_FNAME = os.path.join(data_path, "dict_vec_{}.pkl".format(minimal_frequency))
    if os.path.exists(DICT_VEC_FNAME):
        vec = cPickle.load(open(DICT_VEC_FNAME, "rb"))
        return vec
    else:
        vec = DictVectorizer()
        vec.fit_transform(examples)
        with open(DICT_VEC_FNAME, "wb") as f_out:
            cPickle.dump(vec, f_out)
        return vec


def load_train_dev_test_sentences_other_preprocess(data_path, words_not_to_replace):
    random.seed(123)
    original_train_sents = read_conll_pos_file(os.path.join(data_path, "Penn_Treebank/train.gold.conll"))
    original_dev_sents = read_conll_pos_file(os.path.join(data_path, "Penn_Treebank/dev.gold.conll"))
    original_test_sents = read_conll_pos_file(os.path.join(data_path, "Penn_Treebank/test.gold.conll"))
    tmp_vocab = {w: 1 for w in words_not_to_replace}
    random.shuffle(original_dev_sents)
    random.shuffle(original_train_sents)
    random.shuffle(original_test_sents)
    train_sents = preprocess_sent(tmp_vocab, original_train_sents, 1)
    dev_sents = preprocess_sent(tmp_vocab, original_dev_sents, 1)
    test_sents = preprocess_sent(tmp_vocab, original_test_sents, 1)
    return train_sents, dev_sents, test_sents


def top_words(data_path, minimal_frequency, amount):
    original_train_sents = read_conll_pos_file(os.path.join(data_path, "Penn_Treebank/train.gold.conll"))
    vocab = compute_vocab_count(original_train_sents)
    train_sents = preprocess_sent(vocab, original_train_sents, minimal_frequency)
    count_processed = compute_vocab_count(train_sents)
    return sorted(count_processed.keys(), key=lambda w: count_processed[w])[-amount:]


def preprocessed_train_use_words(data_path, words):
    original_train_sents = read_conll_pos_file(os.path.join(data_path, "Penn_Treebank/train.gold.conll"))
    vocab = {w: 1 for w in words}
    train_sents = preprocess_sent(vocab, original_train_sents, 1)
    return train_sents


def load_train_dev_test_sentences(data_path, minimal_frequency, num_dev_sents=None, num_train_sents=None):
    """
    Returns tagged training, development and test sentences.
    :param data_path:
    :param num_dev_sents:
    :param num_train_sents:
    :param minimal_frequency:
    :return:
    """
    random.seed(123)
    original_train_sents = read_conll_pos_file(os.path.join(data_path, "Penn_Treebank/train.gold.conll"))
    if num_train_sents is not None:
        original_train_sents = original_train_sents[:num_train_sents]
    original_dev_sents = read_conll_pos_file(os.path.join(data_path, "Penn_Treebank/dev.gold.conll"))
    if num_dev_sents is not None:
        original_dev_sents = original_dev_sents[:num_dev_sents]
    original_test_sents = read_conll_pos_file(os.path.join(data_path, "Penn_Treebank/test.gold.conll"))
    random.shuffle(original_dev_sents)
    random.shuffle(original_train_sents)
    random.shuffle(original_test_sents)
    vocab = compute_vocab_count(original_train_sents)
    train_sents = preprocess_sent(vocab, original_train_sents, minimal_frequency)
    dev_sents = preprocess_sent(vocab, original_dev_sents, minimal_frequency)
    test_sents = preprocess_sent(vocab, original_test_sents, minimal_frequency)
    return train_sents, dev_sents, test_sents


def load_data(data_path, num_dev_sents, num_train_sents, minimal_frequency):
    train_sents, dev_sents, test_sents = load_train_dev_test_sentences(data_path, minimal_frequency,
                                                                       num_dev_sents, num_train_sents)
    # The log-linear model training.
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
    dict_vectorizer = get_dict_vectorizer(data_path, train_examples + dev_examples, minimal_frequency)
    print "Vectorize examples"
    train_examples_vectorized = dict_vectorizer.transform(train_examples)
    dev_examples_vectorized = dict_vectorizer.transform(dev_examples)

    return dict_vectorizer, count_word_tags(train_sents), train_examples_vectorized, train_labels, dev_examples_vectorized, dev_labels


def untag_sentence(tagged_sentence):
    return [w for w, t in tagged_sentence]


def get_train_count(data_path):
    train = read_conll_pos_file(os.path.join(data_path, "Penn_Treebank/train.gold.conll"))
    return compute_vocab_count(train)
