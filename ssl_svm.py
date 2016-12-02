#################################################################
# Custom SSL-SVM library
# 
# Daniel Coombs 12/01/16
#
#
##################################################################

from sklearn.neighbors import NearestNeighbors
import numpy as np

class S3VM_SGD:
    """
    """
    def __init__(self, kernel='linear', knn=5, eta0=1.0, alpha=0.001, pest=0.5, prange=0.05):
        self._kernel = kernel
        self._knn = knn
        self._eta0 = eta0 
        self._alpha = alpha
        self._pest = pest
        self._prange = prange
        self._weight_init = False
        self._knn_fit = False
        self._num_train = 0

    def fit(self, X_label, y_label, X_unlabel):
        # TODO: transform data for NLIM

        self._initialize_weights(X_label.shape[1])

        # add column of ones to X for bias
        a = np.ones(len(X_label))
        X_label = np.c_[X_label, a]

        a = np.ones(len(X_unlabel))
        X_unlabel = np.c_[X_unlabel, a]

        # fit labeled data, then unlabeled data
        # TODO: stopping criterion?
        self.fit_label(X_label, y_label)
        self.fit_unlabel(X_label, y_label, X_unlabel)

    def fit_label(self, X, y):
        for x_i, y_i in zip(X, y):
            self._sgd_step(x_i, y_i)

    def fit_unlabel(self, X_label, y_label, X_unlabel):
        if self._knn_fit == False:
            self._fit_knn(X_label)

        num_ones = 0
        num_train = 0
        for x_i in X_unlabel:
            # determine knn
            knn = self._kneighbors(x_i)

            # predict label
            y_i = np.sign(np.sum(y_label[knn]))

            # # PROBLEM: y_i doesn't have information about the boudary, need to use
            # # predictions of a buffer of past unabled points

            # # TODO: fix this
            # # # evaluate p_y < pest
            # num_train += 1
            # if y_i == 1: 
            #     num_ones += 1
            #     p_y =  num_ones / num_train
            # else:
            #     p_y = 1 - num_ones / num_train
            
            # print(knn, y_i, p_y)

            # # # keep in range, if out of range take step
            if p_y < self._pest:
                self._sgd_step(x_i, y_i)

    def decision_function(self, X):
        a = np.ones(len(X))
        X = np.c_[X, a]

        return np.inner(self._weights, X)

    def predict(self, X):
        preds = np.sign(self.decision_function(X))
        preds[preds == 0] = 1
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        error_cnt = np.sum(preds != y)
        return 1 - error_cnt / y.size

    def _initialize_weights(self, n_dims):
        # extra dim for bias term
        self._weights = np.zeros(n_dims + 1)
        self._weight_init = True

    def _fit_knn(self, X):
        # TODO: don't use sklearn
        self._neigh = NearestNeighbors(self._knn + 1)
        self._neigh.fit(X)
        self._knn_fit = True

    def _kneighbors(self, x_i):
        # TODO: don't use sklearn
        _, knn = self._neigh.kneighbors(x_i.reshape(1, -1))
        return knn[0, 1:]

    def _sgd_step(self, x_i, y_i):
        # learning rate decay
        self._lr = self._eta0 / (1 + self._alpha * self._eta0 * self._num_train)

        # update calculation
        if ((self._weights @ x_i) * y_i) <= 1:
            update = y_i * x_i
        else:
            update = 0

        # update
        self._weights = (1 - self._lr * self._alpha) * self._weights \
                        + self._lr * update
        self._num_train += 1