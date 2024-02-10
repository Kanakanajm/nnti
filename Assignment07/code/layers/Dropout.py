import numpy as np
import copy
import random


class Dropout:
    def __init__(self, layer, p: float = 0.5):
        self.p = p
        self.layer = layer

    def __call__(self, x):
        """
        apply inverted dropout
        """
        self.x = self.layer(x)
        # Create an array of random numbers from uniform distribution
        self.mask = (np.random.rand(*self.x.shape) > self.p) / (1 - self.p)
        return self.x * self.mask

    def get_type(self):
        return "layer"

    def grad(self, in_gradient):
        return self.layer.grad(in_gradient)
