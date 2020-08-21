import numpy as np

class Activation(object):
    def __call__(self, input):
        return None

    def derivative(self, input):
        return None

class Sigmoid(Activation):
    def __call__(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        return self(input) * (1 - self(input))

class ReLU(Activation):
    def __call__(self, input):
        return np.maximum(input, 0)

    def derivative(self, input):
        return np.heaviside(input, 0)

class Identity(Activation):
    def __call__(self, input):
        return input

    def derivative(self, input):
        return np.ones(input.shape)

class Softmax(Activation):
    def __call__(self, input):
        return (np.exp(input).T / np.sum(np.exp(input), axis=1)).T

    def derivative(self, input):
        return input * (1 - input)
