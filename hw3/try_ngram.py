import os
import sys
import pandas as pd
import re

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(BASE_DIR, "data")
sys.path.insert(0, ROOT_DIR)

from hw3 import memm
from hw3 import data
import ngram_model

train, dev, test = memm.load_train_dev_test_sentences(DATA_PATH, 50)
train = map(data.untag, train)
dev = map(data.untag, dev)

model = ngram_model.NGramModel(train, 0.25, 0.55)

print 'dev perplexity', model.sentences_perplexity(dev)

sentences = [model.generate_sentence(30) for i in xrange(10)]
for s in sentences:
    print s
print model.sentences_perplexity(sentences)


sys.exit(0)

vocab = pd.read_table(DATA_PATH + "/lm/vocab.ptb.txt", header=None, sep="\s+",
                      index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000

def load_dataset(fname):
    docs = []
    with open(fname) as fd:
        cur = []
        for line in fd:
            # new sentence on -DOCSTART- or blank line
            if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                if len(cur) > 0:
                    docs.append(cur)
                cur = []
            else: # read in tokens
                cur.append(line.strip().split("\t",1))
        # flush running buffer
        docs.append(cur)
    return docs


# Load the training set
docs_train = load_dataset(DATA_PATH + '/lm/ptb-train.txt')
counts = memm.compute_vocab_count(docs_train)
s_train = map(memm.untag, memm.preprocess_sent(counts, docs_train, 40))
print 'docs_train[0]', docs_train[0]

model2 = ngram_model.NGramModel(s_train, 0.33, 0.533)
print model2.sentences_perplexity(s_train)

docs_dev = load_dataset(DATA_PATH + '/lm/ptb-dev.txt')
S_dev = map(memm.untag, memm.preprocess_sent(counts, docs_dev, 40))

for l1 in (0.1, 0.2, 0.3, 0.4):
    for l2 in (0.4, 0.5, 0.6):
        if l1 + l2 < 1:
            print l1, l2
            model2.set_lambda(l1, l2)
            print model2.sentences_perplexity(S_dev)


