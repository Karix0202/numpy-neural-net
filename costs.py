import numpy as np

class Cost(object):
    def __call__(self, y_hat, y_true):
        return None

    def mean(self, y_hat, y_true):
        return None

    def derivative(y_hat, y_true):
        return None


class MSE(Cost):
    def __call__(self, y_hat, y_true):
        return (y_hat - y_true) ** 2

    def mean(self, y_hat, y_true):
        return ((y_hat - y_true) ** 2).mean()

    def derivative(self, y_hat, y_true):
        return 2 * (y_hat - y_true)


class BinaryCrossEntropy(Cost):
    def __call__(self, y_hat, y_true):
        return -np.multiply(y_true, np.log(y_hat)) - np.multiply((1 - y_true), np.log(1 - y_hat))

    def mean(self, y_hat, y_true):
        return (-np.multiply(y_true, np.log(y_hat)) - np.multiply((1 - y_true), np.log(1 - y_hat))).mean()

    def derivative(self, y_hat, y_true):
        return y_hat - y_true
