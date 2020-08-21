from layers import *
from costs import *
from utils import *
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self, cost):
        self.layers = []
        self.training_errors = []

        self.target_labels = None

        self.cost = cost()

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise Exception('Layer must be instance of layers.Layer class')

        self.layers.append(layer)

    def fit(self, X, y, n_iter=1000, eta=.001, target_labels=None, verbose=True, plot=False):
        self.check_layers()
        self.check_shape(X)
        self.check_shape(y)

        self.target_labels = target_labels

        self.pass_args({
            'eta': eta,
        })

        for iter in range(n_iter):
            out = X
            for layer in self.layers:
                out = layer.forward(out)

            self.backward(out, y)
            err = self.cost.mean(self.predict(X), y)
            self.training_errors.append(err)

            if verbose and iter % 10 == 9:
                print('ITER: {}, ERR: {}'.format(iter+1, err))

        self.plot_cost(self.training_errors)

    def predict(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)

        return out

    def pass_args(self, args):
        for layer in self.layers:
            layer.eta = args['eta']

    def check_layers(self):
        if len(self.layers) <= 0:
            raise Exception('Model must have at least one layer')

    def backward(self, output, y_true):
        grad = self.cost.derivative(output, y_true)

        for i in range(len(self.layers))[::-1]:
            grad = self.layers[i].backward(grad)

    def check_shape(self, X):
        if len(X.shape) != 2:
            raise Exception('Data passed should be in shape (n data, k features), but got in shape: {}'.format(X.shape))

    def __call__(self, X):
        self.check_shape(X)

        if isinstance(self.cost, MSE):
            return self.predict(X)
        elif isinstance(self.cost, CrossEntropy):
            if self.target_labels != None:
                return one_hot_then_names(self.predict(X), self.target_labels)
            else:
                return one_hot_encode(self.predict(X))

    def plot_cost(self, errs):
        plt.plot(np.arange(len(errs)), errs)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')

        plt.show()
