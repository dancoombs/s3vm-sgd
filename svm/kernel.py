import numpy as np

class Kernel:
    """ Kernels implemented same as from scikit-learn
    http://scikit-learn.org/stable/modules/svm.html#svm-kernels
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