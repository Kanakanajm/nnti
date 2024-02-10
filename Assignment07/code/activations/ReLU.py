import numpy as np


class ReLU:
    def __init__(self):
        pass

    def __call__(self, x):
        self.x = x
        x[x < 0] = 0
        return x

    def get_type(self):
        return "activation"

    # assign gradient of zero if x = 0 (even though the function is not differentiable at that point)
    def grad(self, in_gradient):
        gradient = self.x.copy()
        gradient[gradient > 0] = 1
        gradient[gradient <= 0] = 0
        return in_gradient * gradient
