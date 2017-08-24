import time
import scipy
from collections import Counter
import numpy as np
import os
import theanets
import cPickle
import random
from hw3 import inputs_generator
from hw3 import model
from hw3 import model_interface
from hw3 import memm
from hw3 import full_information_experiments
np.random.seed(123)
random.seed(123)
train, dev, test = memm.load_train_dev_test_sentences("hw3/data", 20)
c = theanets.Classifier.load(os.path.expanduser("~/Downloads/all_freq20.pkl"))
dict_vectorizer = memm.get_dict_vectorizer("hw3/data/", None, 20)
interface = model_interface.ModelInterface(c, dict_vectorizer)
count = full_information_experiments.count_words(test)
words = random.sample([x for x in count if count[x] > 50], 75)
print 'len(words)', len(words)
gen = inputs_generator.SequentialInputsGenerator(inputs_generator.constant_generator(40), words)
print 'gnerating sentences'
dense_sentences = [gen.generate_input() for _ in xrange(8000)]
probs = [];tagged = []
print 'quering'
for s in dense_sentences:
    p, t = interface.predict_proba(s)
    tagged.append(t)
    probs.append(p)

print 'transform data'
probs, sparse, predictions = full_information_experiments.transform_input_for_training(dict_vectorizer, probs, tagged)


all_i_sorted = sorted(set(sparse.indices))
print 'len(all_i)', len(all_i_sorted)


columns = [sparse.getcol(i).toarray() for i in all_i_sorted]
dense_matrix = np.hstack([column for column in columns])

##
##def null(A, eps=1e-15):
##    u, s, vh = scipy.linalg.svd(A)
##    null_mask = (s <= eps)
##    null_space = scipy.compress(null_mask, vh, axis=0)
##    return scipy.transpose(null_space)

def trange(n):
    start = time.time()
    last_print = start
    for i in xrange(n):
        if time.time() - last_print > 5:
            elapsed = time.time() - start
            left = (n - i) * elapsed / i
            percent = i / float(n) * 100
            last_print = time.time()
            print 'elapsed: {:.1f} left: {:.1f} :: {:.2f}%'.format(elapsed, left, percent)
        yield i

def GEPP(A, eps=1e-5):
    '''
    Gaussian elimination with partial pivoting.
    % input: A is an n x n nonsingular matrix
    '''
    n =  A.shape[0]
    def swap_rows(i, j):
        A[[i, j]] = A[[j, i]]
    # k represents the current pivot row. Since GE traverses the matrix in the upper 
    # right triangle, we also use k for indicating the k-th diagonal column index.
    k = 0
    
    for column in trange(A.shape[1]):
        maxindex = abs(A[k:, column]).argmax() + k
        # print maxindex
        if abs(A[maxindex, column]) < eps:
            continue
        else:
            pass
            # print 'found column=', column
        #Swap rows
        if maxindex != k:
            swap_rows(k, maxindex)
        A[k] /= A[k, column]
        for row in xrange(n):
            if row == k or abs(A[row, column]) < eps:
                continue
            A[row] -= A[row, column] * A[k]
            #Equation solution column
        k += 1
        if k == A.shape[0]:
            break
    A[abs(A) < eps] = 0.
    return k
        
    
d = dense_matrix.copy()
print 'equations', d.shape
print 'eliminating equations matrix'
rank = GEPP(d)
print 'rank', rank
def cut_feature(x):
    if '=' in x:
        return x[:x.index('=')]
    else:
        return x
z = np.zeros(len(dict_vectorizer.vocabulary_))
good_i = [i for i in xrange(rank) if len(d[i].nonzero()[0]) == 1]
print 'nonzero per row count', Counter(Counter(d.nonzero()[0]).values())
z[[all_i_sorted[x] for x in good_i]] = 1.
good_features = dict_vectorizer.inverse_transform([z])[0].keys()
print 'good features count'
print Counter([cut_feature(x) for x in good_features])



