import numpy as np
import copy


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weights = np.random.randn(in_features, out_features) * 0.05
        if bias:
            self.bias = np.random.randn(1, out_features) * 0.05
        else:
            self.bias = np.zeros((1, out_features))

    def __call__(self, x):
        # save the current input (should be a minibatch)
        self.x = x
        return self.x @ self.weights + self.bias

    def get_type(self):
        return "layer"

    def grad(self, in_gradient):
        """
        expects in_gradient of size minibatch_size, out_features
        returns dL/dW (size equal to the size of weight matrix)
                dL/dX (size equal to the size of input matrix)
        """
        return self.x.T @ in_gradient, in_gradient @ self.weights.T
