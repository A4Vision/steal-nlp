import cPickle
import gzip
import numpy as np
import theanets

from hw3 import utils


def convert_int32((x, y)):
    return x, np.array(y, dtype=np.int32)


def transform_data_to_distributions((x, y)):
    z = np.ones((y.shape[0], y.max() + 1)) / 100.
    z[range(y.shape[0]), y] = 1. - 1 / 100. * (y.max())
    return x, z


def train_mnist_example():

    print 'loading'
    train_data, validation_data, test_data = map(transform_data_to_distributions,
                                                 map(utils.convert_to_sparse,
                                                     map(convert_int32, cPickle.load(gzip.open(
                                                         "/home/bugabuga/PycharmProjects/Steal-ML/data/mnist.pkl.gz")))))

    # 1. create a model -- here, a regression model.
    net = theanets.Regressor([theanets.layers.base.Input(size=train_data[0].shape[1], sparse='csr'), 100,
                              (10, 'softmax')], loss=utils.RegressionCrossEntropy(2))

    # 2. train the model.
    for train, valid in net.itertrain(train_data, valid=validation_data, algo='nag', weight_l2=0.1,
                                      learning_rate=1.):
        print('training loss:', train['loss'])
        print('most recent validation loss:', valid['loss'])
        acc = utils.regression_accuracy(net, validation_data[0], np.argmax(validation_data[1], axis=1))
        print 'acc=', acc

    # 3. use the trained model.
    acc = utils.regression_accuracy(net, test_data[0], np.argmax(test_data[1], axis=1))
    print 'acc', acc
