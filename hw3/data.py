import collections
import os
import re

MIN_FREQ = 2
UNKNOWN_WORD = "UNK"

def invert_dict(d):
    res = {}
    for k, v in d.iteritems():
        res[v] = k
    return res


def read_conll_pos_file(path):
    """
        Takes a path to a file and returns a list of word/tag pairs
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                curr.append((tokens[1], tokens[3]))
    return sents


def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1


def untag(tagged_setence):
    return [w for w, t in tagged_setence]


def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    counter = collections.Counter()
    for sentence in sents:
        counter.update(untag(sentence))
    return counter


# Word categories for named-entity recognition table, copied from Bikel et al. (1999)
#   http://people.csail.mit.edu/mcollins/6864/slides/bikel.pdf
# For a specific learning task, we will craft better pseudo-words.

# Word class Example Intuition
# ============================
# twoDigitNum 90 Two digit year
# fourDigitNum 1990 Four digit year
# containsDigitAndAlpha A8956-67 Product code
# containsDigitAndDash 09-96 Date
# containsDigitAndSlash 11/9/89 Date
# containsDigitAndComma 23,000.00 Monetary amount
# containsDigitAndPeriod 1.00 Monetary amount,percentage
# othernum 456789 Other number
# allCaps BBN Organization
# capPeriod M. Person name initial
# firstWord first word of sentence no useful capitalization information
# initCap Sally Capitalized word
# lowercase can Uncapitalized word
# other , Punctuation marks, all other words
SUFFIXES = ('ed', 'es', 'us', 's', 'able', 'ing', 'al', 'ic', 'ly', 'tion')
PREFIXES = ('re', 'dis', 'un', 'de')

CRAFTED_CATEGORIES = (['wordSuffix' + x for x in SUFFIXES]
                      + [x + 'WordPrefix' for x in PREFIXES] +
                      ['withDash',
                       'twoDigitNum', 'fourDigitNum', 'othernum',
                       'containsDigitAndAlpha', 'containsDigitAndDash',
                       'containsDigitAndSlash', 'containsDigitAndPeriod',
                       'allCaps', 'capPeriod', 'firstWord', 'initCap',
                       'lowercase', 'UNK'])


def replace_word(word, is_first, vocab, minimal_fequency):
    """
        Replaces rare words with categories (numbers, dates, etc...)
    """
    ### YOUR CODE HERE
    if word.isdigit():
        if len(word) == 2:
            return 'twoDigitNum'
        elif len(word) == 4:
            return 'fourDigitNum'
        else:
            return 'othernum'
    elif contains_digit(word):
        if contains_alpha(word):
            return 'containsDigitAndAlpha'
        elif '-' in word:
            return 'containsDigitAndDash'
        elif '/' in word or '\\' in word:
            return 'containsDigitAndSlash'
        elif '.' in word:
            return 'containsDigitAndPeriod'
    if word.isalpha() and word.isupper():
        return 'allCaps'
    elif CAP_PERIOD_PATTERN.match(word):
        return 'capPeriod'
    if is_first and vocab.get(word.lower(), 0) >= minimal_fequency:
        return word.lower()
    if not is_first and word[0].isupper():
        return 'initCap'
    if word.isalpha():
        for suffix in SUFFIXES:
            if word.endswith(suffix):
                return 'wordSuffix' + suffix
    if word.isalpha():
        for prefix in PREFIXES:
            if word.startswith(prefix):
                return prefix + 'WordPrefix'
    if '-' in word:
        return 'withDash'
    elif word.isalpha() and word.lower() == word:
        return 'lowercase'
    ### END YOUR CODE
    return UNKNOWN_WORD


CAP_PERIOD_PATTERN = re.compile("^[A-Z]\\.$")
ALPHA_PATTERN = re.compile("[a-zA-Z]")
DIGIT_PATTERN = re.compile("\\d")


def contains_digit(word):
    return DIGIT_PATTERN.search(word) is not None


def contains_alpha(word):
    return ALPHA_PATTERN.search(word) is not None


def preprocess_sent(vocab, sents, minimal_frequency=MIN_FREQ):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        is_first = True
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= minimal_frequency:
                new_sent.append(token)
            else:
                r = replace_word(token[0], is_first, vocab, minimal_frequency)
                assert r in CRAFTED_CATEGORIES or vocab
                new_sent.append((r, token[1]))
                replaced += 1
            total += 1
            is_first = False
        res.append(new_sent)
    print "replaced: " + str(float(replaced) / total)
    return res
