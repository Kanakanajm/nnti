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
        random_array = np.random.uniform(0, 1, self.x.shape)
        # Convert to 1s and 0s based on probability p
        binary_array = (random_array <= self.p).astype(bool)
        self.x[binary_array] = 0
        return self.x

    def get_type(self):
        return "layer"

    def grad(self, in_gradient):
        return self.layer.grad(in_gradient)
