import numpy as np


class ReLU:
    def __init__(self):
        pass

    def __call__(self, x):
        raise NotImplementedError()

    def get_type(self):
        return "activation"

    # assign gradient of zero if x = 0 (even though the function is not differentiable at that point)
    def grad(self, in_gradient):
        raise NotImplementedError()
