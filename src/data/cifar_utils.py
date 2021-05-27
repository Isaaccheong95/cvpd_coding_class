from six.moves import cPickle as pickle
import numpy as np
import os


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:

        datadict = pickle.load(f, encoding="latin1")
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT, train=True):
    """ load all of cifar """
    xs = []
    ys = []

    if train:
        for b in range(1, 6):
            f = os.path.join(ROOT, "data_batch_%d" % (b,))
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)

        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
        return Xtr, Ytr

    else:
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))

        return Xte, Yte
