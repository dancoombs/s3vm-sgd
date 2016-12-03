#################################################################
# Custom SVM library
# 
# Daniel Coombs 12/03/16
#
##################################################################

import numpy as np
import cvxopt
from .kernel import Kernel

class SVM:
    """ Support Vector Machine classifier
    Built to have a similar interface as SVC from scikit-learn

    Parameters
    ----------

    Notes
    -----
    SVM solved via Quadratic Programming
    Implementation uses CVXOPT to solve svm optimization QP problem
    Inspiration taken from Andrew Tulloch 
    http://tullo.ch/articles/svm-py/

    """
    def __init__(self, C=1.0, kernel='linear', min_weight=1e-5, 
                 gamma=0.5, degree=3, coef0=0.0):
        self.C = C
        self.min_weight = min_weight
        
        if gamma <= 0:
            print('Gamma must be greater than 0, defualt to 0.5')
            gamma = 0.5

        self._gamma = gamma
        self._degree = degree
        self._coef0 = coef0
        
        if kernel == 'linear':
            self._kernel = Kernel.linear()
        elif kernel == 'rbf':
            self._kernel = Kernel.rbf(self._gamma)
        elif kernel == 'poly':
            self._kernel = Kernel.poly(self._degree, self._coef0)
        # elif kernel == 'sigmoid':
        #     self._kernel = Kernel.sigmoid(self._coef0)
        else:
            print('Invalid Kernel, defualt to linear')
            self._kernel = Kernel.linear()

    def fit(self, X, y):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        sol_weights = self._solve_dual(X, y)

        # select support vectors from minimum weight
        weight_inds = sol_weights  > self.min_weight
        self._weights = sol_weights[weight_inds]
        self._support_vectors = X[weight_inds]
        self._support_vector_labels = y[weight_inds]

        # solve for bias term by averaging error over support vectors w/ bias 0
        self.bias = 0
        self.bias = np.mean([y_k - self.decision_function([x_k])
                            for(y_k, x_k) in zip(self._support_vector_labels, self._support_vectors)])

    def decision_function(self, X):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        result = self.bias * np.ones(len(X))
        # Loop around each sample of X
        for i in range(0, len(X)):
            # Sum result from each support vecor for each sample
            for j in range(0, len(self._weights)):
                result[i] += self._weights[j] * self._support_vector_labels[j] \
                          * self._kernel(self._support_vectors[j], X[i])
       
        return result
        
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

    def _compute_kernel_matrix(self, X):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        n_samples = X.shape[0]
        K = np.zeros([n_samples, n_samples])
        for i in range(0, n_samples):
            for j in range(0, n_samples):
                # hack to get single points to work with rbf
                K[i, j] = self._kernel(X[i], X[j])
        return K

    def _solve_dual(self, X, y):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        n_samples, n_features = X.shape
        K = self._compute_kernel_matrix(X)
        # Solve lagrangian dual
        # Minimize x'Px + q'x
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # subject to Gx <= h
        # x > 0 -> -x < 0
        G_std = -1*np.identity(n_samples)
        h_std = np.zeros((n_samples,1))
        # x < C
        G_slack = np.identity(n_samples)
        h_slack = np.ones((n_samples,1)) * self.C
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        # subject to Ax = b
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0.0)

        # solve
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(sol['x'])


class SVM_SGD:
    """ Support Vector Machine classifiers
    Built to have a similar interface as SVC from scikit-learn
    Trained using stochastic gradient decent

    Parameters
    ----------

    Notes
    -----

    """
    def __init__(self, eta0=1.0, alpha=0.0001):
        self._eta0 = eta0
        self._alpha = alpha
        self._weight_init = False

    def fit(self, X, y, batch_size=1):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        # split into batches & train on each batch
        num_batch = int(len(X) / batch_size)
        X_batches = np.array_split(X, num_batch)
        y_batches = np.array_split(y, num_batch)
        
        self._initialize_weights(X.shape[1])

        for i in range(0, len(X_batches)):
            # learning rate decay??
            self._lr = self._eta0 / (1 + self._alpha * self._eta0 * i) 
            self.fit_partial(X_batches[i], y_batches[i])

    def fit_partial(self, X, y):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        if self._weight_init == False:
            self._initialize_weights(X[0].shape[0])

        # add column of ones to X for bias
        a = np.ones(len(X))
        X = np.c_[X, a]

        # Sum gradient update for each sample in batch
        update = 0
        for x_i, y_i in zip(X, y):
            if ((self._weights @ x_i) * y_i) <= 1:
                update += y_i * x_i

        self._weights = (1 - self._lr * self._alpha) * self._weights \
                        + self._lr / len(X) * update

    def decision_function(self, X):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        # add column of ones to X for bias
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

    def fit_batch_score(self, X_train, y_train, X_test, y_test, batch_size=1):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        # split into batches & train on each batch
        num_batch = int(len(X_train) / batch_size)
        X_batches = np.array_split(X_train, num_batch)
        y_batches = np.array_split(y_train, num_batch)
        
        self._initialize_weights(X_train.shape[1])

        scores = np.zeros(len(X_batches))
        lrates = np.zeros(len(X_batches))
        for i in range(0, len(X_batches)):
            # learning rate decay??
            self._lr = self._eta0 / (1 + self._alpha * self._eta0 * i)
            self.fit_partial(X_batches[i], y_batches[i])
            scores[i] = self.score(X_test, y_test)
            lrates[i] = self._lr

        return scores, lrates

    def _initialize_weights(self, n_dims):
        """
        

        Parameters
        ----------

        Returns
        -------
        """
        # extra dim for bias term
        self._weights = np.zeros(n_dims + 1)
        self._weight_init = True

