####################################################################
# Custom Online Semi-Supervised SVM
# 
# Daniel Coombs 12/03/16
#
# Done as a course project for ECE598NS "Machine Learning In Silicon" 
# at the University of Illinos at Urbana-Champaign
#
#####################################################################

from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures
from collections import deque
import numpy as np

# helper functions for S3VM SGD
def _get_knn(k, X, x_i):
    """
    Calculates the K-nearest-neighbors of a given sample x_i by finding the K
    closest samples in X in terms of their euclidean distances from the given
    sample.

    Parameters
    ----------
    X : vector of samples from which to search for knn

    x_i : sample to find knn for

    Returns
    -------
    knn : vector of indicies in X of the k closest samples. 
    """

    # calcualte euclidean distance
    dist = np.sum((X - x_i)**2, axis=1)**0.5
    
    # return k smallest distances
    knn = np.argsort(dist)[:k]
    return knn


def _transform(X, kernel, gamma=1.0):
    """
    Transfroms the input samples for a kernel input mapping. If the specified
    kernel is non-linear this can be called a NLIM (non-linear input mapping)

    Parameters
    ----------
    X : vector of input samples

    kernel : kernel to perform the transformation

    gamma : optional (default=1.0)
        Only relevant to RBF kernel. Must be greater than 0.
        Gamma parameter for RBF function.

    Returns
    -------
    X_tran : vector of transformed samples
    
    """

    # TODO: custom implementations of these for silicon
    if kernel == 'linear':
        X_tran = X
    elif kernel == 'rbf':
        if gamma < 0:
            print('Gamma must be greater than 0: setting to default')
            gamma = 1.0

        rbf_feature = RBFSampler(gamma=gamma, n_components=n, random_state=1)
        X_tran = rbf_feature.fit_transform(x)
    else:
        print('Kernel not supported, defaulting to linear')
        X_tran = X

    return X_tran


class S3VM_SGD:
    """
    Semi-supervised support vector machine
    Performs online stochastic gradient decent using large scale manifold transduction
    as described in Karlen et. al [1].

    Parameters
    ----------
    kernel : optional (default='linear')
        Kernel name for the input mapping

    knn : optional (default=5)
        K-nearest-neighbors parameter for the amount of neighbors to use in
        majority vote for determining the label for an unlabeled sample

    eta0 : optional (default=1.0)
        Initial learning rate

    alpha : optional (default=0.001)
        Learning rate decay factor

    pest : optional (default=0.5)
        Estimation for the percentage of unlabled samples that should be assigned
        to class 1 for use ing the balancing constraint 

    buffer_size : optional (default=25)
        Buffer size for unlabeled samples and their assigned labels for use in 
        the balancing constraint

    gamma : optional (default=1.0)
        Only relevant for RBF kernel. Gamma parameter for RBF function

    Notes
    -----

    References
    ----------
    [1] http://ronan.collobert.com/pub/matos/2008_transduction_icml.pdf

    """

    def __init__(self, kernel='linear', knn=5, eta0=1.0, alpha=0.001, pest=0.5, buffer_size=25,
                    gamma=1.0):
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
        Performs semi-supervised learning using the labled and unlabeled input data.
        First performs an input mapping, then trains the SVM using SGD. The labled
        data is used to create an initial classifier, then the unlabeled data is used
        to iterate on that classifier using a method similar to that descibed in [1].

        Parameters
        ----------
        X_label : vector of samples of labled data

        y_label : lables corresponding to X_label

        X_unlabel : vector of samples of unlabled data

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
        Fits the labled data by taking a gradient decent step for each sample

        Parameters
        ----------
        X : vector of samples of labeled data

        y : labels corresponding to X
        """

        for x_i, y_i in zip(X, y):
            self._sgd_step(x_i, y_i)

    def fit_unlabel(self, X_label, y_label, X_unlabel):
        """
        Preforms online semi-supervised learning on the unlabled samples by
        estimating its label by a majority vote of neighbors. The algorithm
        then uses that sample and its label to make a gradient decent step.
        A balancing constraint is utilized to attempt to keep the amount of
        unlabled samples labled as a 1 equal to the estimation parameter pest

        Parameters
        ----------
        X_label : vector of samples of labeled data

        y_label : labels corresponding to X_label

        X_unlabel : vector of samples of unlabled data

        Notes
        -----
        At the moment this function guesses the label of the unlabled
        sample by taking a majority vote of its k nearest labled samples
        and always makes an SGD step in that direction

        TODO: Implement a balancing constraint
        """

        # initialize buffer
        # buff_x = deque(maxlen=self._buffer_size)
        # buff_y = deque(maxlen=self._buffer_size)

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
        Computes numerical value of the decision function. This sign of this value indicates which
        side of the decision surface the sample is on. The magnitude determines how far away the
        sample is from the decision service.

        Parameters
        ----------
        X : vector of samples


        Returns
        -------
        result : vector of values of the decision function coressponding to X

        """

        a = np.ones(len(X))
        X = np.c_[X, a]
        return np.inner(self._weights, X)

    def predict(self, X):
        """
        Predicts the class for each sample in X using the sign of the decision function. If samples
        are directly on the margin (decision function = 0) it uses the convention to assign them
        to class 1

        Parameters
        ----------
        X : vector of samples to be labeled

        Returns
        -------
        preds : predicted labels corresponding to X
        """

        preds = np.sign(self.decision_function(X))
        preds[preds == 0] = 1
        return preds

    def score(self, X, y):
        """
        Function for testing the percent error of the classifier. Takes a set of labeled data and 
        compares the predicted labels to its set of known labels. Returns the percent of correctly
        predicted labels.

        Parameters
        ----------
        X : test vector

        y : target labels corresponding to X

        Returns
        -------
        Sucess percentage
        """

        preds = self.predict(X)
        error_cnt = np.sum(preds != y)
        return 1 - error_cnt / y.size

    def _sgd_step(self, x_i, y_i):
        """
        Takes a single stochastic gradient decent step corresponding to the input
        sample

        Parameters
        ----------
        x_i : input feature vector

        y_i : label corresponding to feature vector x_i
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
        Creates the weights vector and initializes to 0 for SGD

        Parameters
        ----------
        n_dims : number of dimensions for the weights
        
        TODO: is all 0s the best place to start?
        """

        self._weights = np.zeros(n_dims)
        self._weight_init = True

