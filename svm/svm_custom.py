#################################################################
# Custom SVM library
# Contains an SVM solved via quadratic programming and an SVM
# solved via stochastic gradient decent.
#
# Daniel Coombs 12/03/16
#
# Not intended for outside use, done as an exploration into
# different optimization techniques for SVMs
#
##################################################################

import numpy as np
import cvxopt

class SVM:
    """ 
    Support Vector Machine classifier for binary classification
    Solved via Quadratic Programming
    Implementation uses CVXOPT to solve SVM optimization QP problem
    Built to have a similar interface as SVC from scikit-learn
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    Parameters
    ----------
    C : optional (default=1.0)
        Penalty parameter for the soft-margin slack

    kernel : optional (default='linear')
        Kernel mapping function for SVM "kernel-trick"

    min_weight : optional (default=1e-5)
        Minimum weight for sample to be returned as a support vector

    gamma : optional (default=0.5)
        Only relevant to RBF kernel. Must be greater than 0. 
        Gamma parameter for RBF kernel. See reference

    degree : optional (default=3)
        Only relevant to polynomial kernel. Degree of polynomial

    coef0 : optional (default=0.0)
        Only relevenat to polynomial kernel. Offset for inhomogeneous mapping

    Notes
    -----
    
    References
    ----------
    Inspiration taken from Andrew Tulloch: 
    http://tullo.ch/articles/svm-py/
    https://en.wikipedia.org/wiki/Support_vector_machine

    """

    def __init__(self, C=1.0, kernel='linear', min_weight=1e-5, 
                 gamma=0.5, degree=3, coef0=0.0):
        self.C = C
        self.min_weight = min_weight
        
        if gamma <= 0:
            print('Gamma must be greater than 0, defualt to 0.5')
            gamma = 0.5

        self._gamma = gamma

        if degree < 1:
            print('Degree must be >= 1, default to 3')
            degree = 3

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
        Fits the model according to the given training data.

        Parameters
        ----------
        X : training vector who's shape is [n_samples, n_features]

        y : training labels relative to X

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

    def _compute_kernel_matrix(self, X):
        """
        Helper function to compute the kernel matrix for use in QP

        Parameters
        ----------
        X : training vector of size [n_sampels, n_features]

        Returns
        -------
        K : matrix of size [n_samples, n_samples]
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
        Uses CVXOPT's quadtratic programming solver to solve the lagrangian dual formulated
        by soft-margin SVM theory.

        Parameters
        ----------
        X : training vector who's shape is [n_samples, n_features]

        y : training labels relative to X

        Returns
        -------
        Solution : weight value corresponding to each sample in X
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
    """ 
    Support Vector Machine classifier for binary classification
    Trained using stochastic gradient decent using an algorithm similar to
    Pegasos from Shalev-Shwartz et al. but without the projection step

    Built to have a similar interface as SVC from scikit-learn
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    Parameters
    ----------
    eta0 : optional (default=1.0)
        Initial learning rate. Learning rate decays by eta0 / (1 + eta0 * alpha * i)
        where i is the step number

    alpha : optional (default=0.0001)
        Parameter controling the speed at which the learning rate decays

    References
    -----
    http://leon.bottou.org/projects/sgd
    http://www.machinelearning.org/proceedings/icml2007/papers/587.pdf

    """

    def __init__(self, eta0=1.0, alpha=0.0001):
        self._eta0 = eta0
        self._alpha = alpha
        self._weight_init = False

    def fit(self, X, y, batch_size=1):
        """
        Fits the model according to the given training data.

        Parameters
        ----------
        X : training vector who's shape is [n_samples, n_features]

        y : training labels relative to X

        batch_size : optional (default=1)
            Controls the batch size for batch stochastic gradient decent

        """

        # split into batches & train on each batch
        num_batch = int(len(X) / batch_size)
        X_batches = np.array_split(X, num_batch)
        y_batches = np.array_split(y, num_batch)
        
        # extra dim for the bias term added in fit_partial
        self._initialize_weights(X.shape[1] + 1)

        for i in range(0, len(X_batches)):
            # learning rate decay
            self._lr = self._eta0 / (1 + self._alpha * self._eta0 * i) 
            self.fit_partial(X_batches[i], y_batches[i])

    def fit_partial(self, X, y):
        """
        Takes a gradient decent step for the input data

        Parameters
        ----------
        X : batch training vector who's shape is [n_samples, n_features]

        y : batch training labels relative to X

        """
        if self._weight_init == False:
            # extra dim for the bias term
            self._initialize_weights(X[0].shape[0] + 1)

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

        # add column of ones to X for bias
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

    def fit_batch_score(self, X_train, y_train, X_test, y_test, batch_size=1):
        """
        See fits the data using batch SGD. After each batch the score is caluculated on
        the test set for visualization. The learning rate decay is also saved.

        Parameters
        ----------
        X_train : vector of training samples

        y_train : vector of labels corresponding to training samples

        X_test : vector of test samples

        y_test : vector of labels corresponding to test samples

        batch_size : optional (default=1.0)

        Returns
        -------
        scores : scores calculated after each step

        lrates : learning rate for each step
        """

        # split into batches & train on each batch
        num_batch = int(len(X_train) / batch_size)
        X_batches = np.array_split(X_train, num_batch)
        y_batches = np.array_split(y_train, num_batch)
        
        self._initialize_weights(X_train.shape[1])

        scores = np.zeros(len(X_batches))
        lrates = np.zeros(len(X_batches))
        for i in range(0, len(X_batches)):
            # learning rate decay
            self._lr = self._eta0 / (1 + self._alpha * self._eta0 * i)
            self.fit_partial(X_batches[i], y_batches[i])
            scores[i] = self.score(X_test, y_test)
            lrates[i] = self._lr

        return scores, lrates

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


class Kernel:
    """ Kernels implemented same as from scikit-learn
    http://scikit-learn.org/stable/modules/svm.html#svm-kernels

    TODO: Fix sigmoid
    """
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def rbf(gamma):
        return lambda x, y: np.exp(-gamma * np.linalg.norm(np.subtract(x, y)) ** 2)

    @staticmethod
    def poly(degree, r):
        return lambda x, y: (np.inner(x,y) + r) ** degree

    # @staticmethod
    # def sigmoid(r):
    #     return lambda x, y: np.tanh(np.inner(x,y) + r)