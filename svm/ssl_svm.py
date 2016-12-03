#################################################################
# Custom SSL-SVM library
# 
# Daniel Coombs 12/01/16
#
#
##################################################################

from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures
from collections import deque
import numpy as np

# helper functions for S3VM SGD
def _get_knn(k, X, x_i):
    """
    

    Parameters
    ----------

    Returns
    -------
    """
    # knn using euclidean distance
    dist = np.sum((X - x_i)**2, axis=1)**0.5
    knn = np.argsort(dist)[:k]
    return knn


def _transform(X, kernel):
    """
    

    Parameters
    ----------

    Returns
    -------
    """
    # TODO: custom implementations of these for silicon
    if kernel == 'linear':
        X_tran = X
    elif kernel == 'rbf':
        rbf_feature = RBFSampler(gamma=gamma, n_components=n, random_state=1)
        X_tran = rbf_feature.fit_transform(x)
    else:
        print('Kernel not supported, defaulting to linear')
        X_tran = X

    return X_tran


class S3VM_SGD:
    """

    Parameters
    ----------


    Attributes
    ----------


    Notes
    -----

    References
    ----------

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
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        # Transform data for NLIM, add feature for bias
        #X_label, X_unlabel = _transform([X_label, X_unlabel], self._kernel)

        # add column of ones to X for bias
        a = np.ones(len(X_label))
        X_label = np.c_[X_label, a]

        a = np.ones(len(X_unlabel))
        X_unlabel = np.c_[X_unlabel, a]

        # Initialize weights for SGD
        self._initialize_weights(X_label.shape[1])

        # fit labeled data, then unlabeled data
        # TODO: stopping criterion?
        self.fit_label(X_label, y_label)
        self.fit_unlabel(X_label, y_label, X_unlabel)

    def fit_label(self, X, y):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        for x_i, y_i in zip(X, y):
            self._sgd_step(x_i, y_i)

    def fit_unlabel(self, X_label, y_label, X_unlabel):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        # initialize buffer
        buff_x = deque(maxlen=self._buffer_size)
        buff_y = deque(maxlen=self._buffer_size)

        # No balancing constraint, using labels as knn
        for x_i in X_unlabel:
            knn_idx = _get_knn(self._knn, X_label, x_i)
            y_i = np.sign(np.sum(y_label[knn_idx]))
            self._sgd_step(x_i, y_i)

        # With balancing constraint
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
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        a = np.ones(len(X))
        X = np.c_[X, a]
        return np.inner(self._weights, X)

    def predict(self, X):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        preds = np.sign(self.decision_function(X))
        preds[preds == 0] = 1
        return preds

    def score(self, X, y):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        preds = self.predict(X)
        error_cnt = np.sum(preds != y)
        return 1 - error_cnt / y.size

    def _sgd_step(self, x_i, y_i):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
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

    def _initialize_weights(self, n_dims):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        # TODO: better starting point?
        self._weights = np.zeros(n_dims)
        self._weight_init = True

