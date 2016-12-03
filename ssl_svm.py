#################################################################
# Custom SSL-SVM library
# 
# Daniel Coombs 12/01/16
#
#
##################################################################

from sklearn.neighbors import NearestNeighbors
from collections import deque
import numpy as np

class S3VM_SGD:
    """
    """
    def __init__(self, kernel='linear', knn=5, eta0=1.0, alpha=0.001, pest=0.5, buffer_size=25):
        self._kernel = kernel
        self._knn = knn
        self._eta0 = eta0 
        self._alpha = alpha
        self._pest = pest
        self._buffer_size = buffer_size
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
        # initialize buffer
        buff_x = deque(maxlen=self._buffer_size)
        buff_y = deque(maxlen=self._buffer_size)

        # No balancing constraint, using labels as knn
        for x_i in X_unlabel:
            knn_idx = self._get_knn(X_label, x_i)
            y_i = np.sign(np.sum(y_label[knn_idx]))
            self._sgd_step(x_i, y_i)

        # for x_i in X_unlabel:
        #     # fill up buffer before using balancing constraint
        #     if len(buff_x) < 25:
        #         # get knn with labeled data
        #         knn_idx = self._get_knn(X_label, x_i)
                
        #         # predict y_i based on knn from labels
        #         y_i = np.sign(np.sum(y_label[knn_idx]))
        #         self._sgd_step(x_i, y_i)

        #     else:
        #         # get knn with buffer
        #         knn_idx = self._get_knn(buff_x, x_i[:-1])

        #         # calc y_i based on majority vote
        #         buff_preds = self.predict(np.array(list(buff_x))[knn_idx])
        #         y_i = np.sign(np.sum(buff_preds))

        #         # calculate fraction of resenct labels
        #         # TODO: just use a running counter
        #         p_one = buff_y.count(1) / len(buff_y)
        #         print(y_i, p_one)

        #         if y_i == 1 and p_one < self._pest:
        #             self._sgd_step(x_i, y_i)
        #         elif y_i == -1 and p_one > self._pest:
        #             self._sgd_step(x_i, y_i)
            
        #         buff_x.popleft()
        #         buff_y.popleft()
            
        #     # leave out bias
        #     buff_x.append(x_i[:-1])
        #     buff_y.append(y_i)

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

    def _get_knn(self, X, x_i):
        # TODO: don't use sklearn, check this
        knn = NearestNeighbors(self._knn + 1)
        knn.fit(X)
        _, knn_idx = knn.kneighbors(x_i.reshape(1, -1))
        return knn_idx[0, 1:]

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